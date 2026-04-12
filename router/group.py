"""
Group routing: sequential and load-balance strategies.

Groups can contain APIEndpoints or other Groups (nested).
"""
from __future__ import annotations

import random
from typing import Any, AsyncIterator, Dict, List, Tuple, Union

import httpx

from router.config import GroupConfig, RouterConfig
from router.endpoint import APIEndpoint, EndpointUnavailableError, UpstreamError
from router.logger import get_logger

log = get_logger("group")

Member = Union[APIEndpoint, "Group"]


class Group:
    def __init__(self, group_id: str, cfg: GroupConfig, members: List[Member]) -> None:
        self.group_id = group_id
        self.cfg = cfg
        self.members = members

    async def call(
        self,
        request_body: Dict[str, Any],
        request_format: str,
        client: httpx.AsyncClient,
    ) -> Tuple[bool, Any]:
        """
        Route the request through this group's strategy.
        Returns (is_streaming, result).
        """
        if self.cfg.strategy == "sequential":
            return await self._call_sequential(request_body, request_format, client)
        else:
            return await self._call_load_balance(request_body, request_format, client)

    async def _call_sequential(
        self,
        request_body: Dict,
        request_format: str,
        client: httpx.AsyncClient,
    ) -> Tuple[bool, Any]:
        """Try members in order, skip unavailable ones."""
        last_error = None
        for member in self.members:
            available, reason = await _check_available(member)
            if not available:
                log.info(
                    "[%s] skipping member '%s': %s",
                    self.group_id, _member_id(member), reason,
                )
                continue
            try:
                result = await member.call(request_body, request_format, client)
                return result
            except (EndpointUnavailableError, UpstreamError) as e:
                log.warning(
                    "[%s] member '%s' failed: %s", self.group_id, _member_id(member), e
                )
                last_error = e

        raise RoutingError(
            f"[{self.group_id}] all sequential members failed or unavailable: {last_error}"
        )

    async def _call_load_balance(
        self,
        request_body: Dict,
        request_format: str,
        client: httpx.AsyncClient,
    ) -> Tuple[bool, Any]:
        """
        Select a member by weighted random (weights from RPM config),
        fall back to available members if selected is unavailable.
        """
        available_members = []
        weights = []
        for member in self.members:
            ok, reason = await _check_available(member)
            if ok:
                available_members.append(member)
                weights.append(_member_weight(member))

        if not available_members:
            raise RoutingError(
                f"[{self.group_id}] no available members for load balancing"
            )

        # Weighted random selection
        selected = random.choices(available_members, weights=weights, k=1)[0]
        log.info(
            "[%s] load_balance selected '%s'", self.group_id, _member_id(selected)
        )
        try:
            return await selected.call(request_body, request_format, client)
        except (EndpointUnavailableError, UpstreamError) as e:
            log.warning(
                "[%s] selected member '%s' failed, trying others: %s",
                self.group_id, _member_id(selected), e,
            )
            # Fall back to other available members
            for member in available_members:
                if member is selected:
                    continue
                try:
                    return await member.call(request_body, request_format, client)
                except (EndpointUnavailableError, UpstreamError) as e2:
                    log.warning(
                        "[%s] fallback member '%s' also failed: %s",
                        self.group_id, _member_id(member), e2,
                    )
            raise RoutingError(f"[{self.group_id}] all load_balance members failed")

    def stats(self) -> dict:
        return {
            "group_id": self.group_id,
            "strategy": self.cfg.strategy,
            "members": [m.stats() if hasattr(m, "stats") else str(m) for m in self.members],
        }


async def _check_available(member: Member) -> Tuple[bool, str]:
    if isinstance(member, APIEndpoint):
        return await member.is_available()
    # Group: available if at least one member is available
    for m in member.members:
        ok, _ = await _check_available(m)
        if ok:
            return True, ""
    return False, "all sub-members unavailable"


def _member_id(member: Member) -> str:
    if isinstance(member, APIEndpoint):
        return member.api_id
    return member.group_id


def _member_weight(member: Member) -> float:
    """Weight for load balancing (based on RPM if available, else 1.0)."""
    if isinstance(member, APIEndpoint):
        return float(member.cfg.usage.rpm or 1)
    # For groups, sum weights of available sub-members
    return sum(_member_weight(m) for m in member.members) or 1.0


def build_routing_tree(config: RouterConfig) -> Dict[str, Union[APIEndpoint, Group]]:
    """Build endpoint and group objects from config."""
    endpoints: Dict[str, APIEndpoint] = {
        api_id: APIEndpoint(api_id, api_cfg)
        for api_id, api_cfg in config.apis.items()
    }

    groups: Dict[str, Group] = {}

    # Resolve groups in dependency order
    unresolved = dict(config.groups)
    max_passes = len(unresolved) + 1

    for _ in range(max_passes):
        if not unresolved:
            break
        for gid, gcfg in list(unresolved.items()):
            members: List[Member] = []
            resolved = True
            for m in gcfg.members:
                if m.api:
                    members.append(endpoints[m.api])
                elif m.group:
                    if m.group not in groups:
                        resolved = False
                        break
                    members.append(groups[m.group])
            if resolved:
                groups[gid] = Group(gid, gcfg, members)
                del unresolved[gid]

    if unresolved:
        raise ValueError(f"Circular group references: {list(unresolved)}")

    all_nodes: Dict[str, Union[APIEndpoint, Group]] = {}
    all_nodes.update(endpoints)
    all_nodes.update(groups)
    return all_nodes


class RoutingError(Exception):
    pass
