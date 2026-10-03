"""One address grammar for everything a world offers by name: endpoints and frames.

A path is ``<component>/<name>``. The component is the dotted address of the entry that offers the
name, with its dots as slashes -- ``robot.oakd`` is ``robot/oakd`` -- as deep as the leading segments
name a component; the rest is the name that component offers. For an endpoint that is its name
(``robot/lidar/scan``, ``ur5e/force_torque/tare``); for a frame, one of the component's bodies,
sites, cameras, declared frames or device frames, as TF shows it (``robot/base_link``,
``robot/oakd/oakd_rgb_camera_optical_frame``, :mod:`roqsim.frames`). An endpoint is also named by
the entity that owns it and its name, where that entity has one of that name. An offer with no
component is named by its name alone: a frame of the world's own MJCF that no entity owns
(``gantry``).

A caller says which kind it wants, so an endpoint and a frame of one name never collide. Within a
kind, a path that names two offers (a body and a site of one name) is refused naming both, and an
unknown one names the nearest known path and what its component offers. This module holds the
matching only; :func:`roqsim.frames.resolve_frame` builds the frame offers, and a scenario's
endpoint lookup the endpoint offers, each from what it can see.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .document import nearest


@dataclass(frozen=True)
class Offer:
    """One thing a path can name.

    Attributes:
        component: the offering component's path (``robot/oakd``)
        name: what the component calls it, which may contain slashes (``bumper/bump_left``)
        kind: what the caller asks for: ``frame``, or an endpoint's direction
        what: the offer's own sort, for a refusal (``body``, ``site``, ``command``)
        alias: a second path that names it: an endpoint's owner and name
        target: what the caller resolves it to
    """

    component: str
    name: str
    kind: str
    what: str = ""
    alias: str = ""
    target: Any = field(default=None, compare=False, repr=False)

    @property
    def path(self) -> str:
        return f"{self.component}/{self.name}" if self.component else self.name


class PathError(LookupError):
    """A path that names nothing of the kind asked for, or more than one thing.

    ``reason`` is ``unknown`` or ``ambiguous``; ``matches`` the offers an ambiguous path names;
    ``suggestion`` the nearest known path; ``component`` the deepest component the path names and
    ``offered`` the paths it offers of that kind.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        matches: tuple[Offer, ...] = (),
        suggestion: str | None = None,
        component: str = "",
        offered: tuple[str, ...] = (),
    ):
        super().__init__(message)
        self.reason = reason
        self.matches = matches
        self.suggestion = suggestion
        self.component = component
        self.offered = offered


def address_path(address: str) -> str:
    """A dotted component address as a path: ``robot.oakd`` is ``robot/oakd``."""
    return address.replace(".", "/")


def resolve(offers: Iterable[Offer], path: str, kind: str, *, noun: str | None = None) -> Offer:
    """The one offer of *kind* that *path* names, or :class:`PathError`.

    *noun* words the refusal (``frame``, ``command``); it defaults to *kind*.
    """
    noun = noun or kind
    path = path.strip("/")
    pool = [o for o in offers if o.kind == kind]
    matches = [o for o in pool if o.path == path or (o.alias and o.alias == path)]
    unique = list({(o.path, o.what): o for o in matches}.values())
    if len(unique) == 1:
        return unique[0]
    if unique:
        named = ", ".join(f"{o.path} ({o.what})" if o.what else o.path for o in unique)
        raise PathError(
            f"{path!r} names {len(unique)} {noun}s: {named}.",
            reason="ambiguous",
            matches=tuple(unique),
        )
    known = {o.path for o in pool} | {o.alias for o in pool if o.alias}
    suggestion = nearest(path, known)
    components = {o.component for o in pool}
    segments = path.split("/")
    component = ""
    for depth in range(len(segments) - 1, 0, -1):
        prefix = "/".join(segments[:depth])
        if prefix in components:
            component = prefix
            break
    offered = tuple(sorted(o.path for o in pool if o.component == component)) if component else ()
    message = f"no {noun} {path!r}."
    if suggestion:
        message += f" Did you mean {suggestion!r}?"
    if component:
        listed = ", ".join(offered[:40]) + (" ..." if len(offered) > 40 else "")
        message += f" {component!r} offers: {listed}."
    else:
        tops = sorted({c.split("/")[0] for c in components if c})
        message += f" The components offering one: {', '.join(tops) or '(none)'}."
        top_level = sorted({o.path for o in pool if not o.component and o.path not in components})
        if top_level:
            listed = ", ".join(top_level[:40]) + (" ..." if len(top_level) > 40 else "")
            message += f" Named at the top of the world: {listed}."
    raise PathError(
        message,
        reason="unknown",
        suggestion=suggestion,
        component=component,
        offered=offered,
    )
