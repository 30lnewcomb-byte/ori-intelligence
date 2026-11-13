"""Pure-Python validation and OpenSCAD compiler for constrained CAD plans."""

from __future__ import annotations

import math
from typing import Any

MAX_OPERATIONS = 128
MAX_MESH_VERTICES = 5000
MAX_MESH_FACES = 10000
MAX_NESTING_DEPTH = 16


def _is_finite_number(value: Any) -> bool:
    """Reject booleans, NaN, and infinity as geometry measurements."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def compile_cad_plan(plan: dict[str, Any]) -> str:
    """Compile a validated parametric CAD plan into OpenSCAD source."""
    if not isinstance(plan, dict):
        raise ValueError("CAD plan must be an object.")

    operations = plan.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("CAD plan must contain a non-empty operations list.")
    if len(operations) > MAX_OPERATIONS:
        raise ValueError(f"CAD plan cannot contain more than {MAX_OPERATIONS} operations.")

    def render_shape(op: dict[str, Any], indent: str = "", depth: int = 0) -> list[str]:
        if depth > MAX_NESTING_DEPTH:
            raise ValueError("CAD plan nesting is too deep.")

        kind = op.get("kind")

        if kind == "cube":
            size = op.get("size")
            center = op.get("center", True)
            if not (
                isinstance(size, list)
                and len(size) == 3
                and all(_is_finite_number(value) and value > 0 for value in size)
            ):
                raise ValueError("cube.size must contain three finite positive numbers.")
            if not isinstance(center, bool):
                raise ValueError("cube.center must be true or false.")
            return [
                f"{indent}cube([{size[0]}, {size[1]}, {size[2]}], center={str(center).lower()});"
            ]

        if kind == "cylinder":
            radius = op.get("radius")
            height = op.get("height")
            center = op.get("center", True)
            if not (
                _is_finite_number(radius)
                and radius > 0
                and _is_finite_number(height)
                and height > 0
            ):
                raise ValueError("cylinder requires finite positive radius and height.")
            if not isinstance(center, bool):
                raise ValueError("cylinder.center must be true or false.")
            return [
                f"{indent}cylinder(r={radius}, h={height}, center={str(center).lower()});"
            ]

        if kind in {"translate", "rotate"}:
            value = op.get("value")
            child = op.get("shape")
            if not (
                isinstance(value, list)
                and len(value) == 3
                and all(_is_finite_number(component) for component in value)
                and isinstance(child, dict)
            ):
                raise ValueError(f"{kind} requires a three-number value and a shape object.")
            body = render_shape(child, indent + "  ", depth + 1)
            return [
                f"{indent}{kind}([{value[0]}, {value[1]}, {value[2]}]) {{",
                *body,
                f"{indent}}}",
            ]

        if kind == "mesh":
            vertices = op.get("vertices")
            triangles = op.get("triangles")
            if not (
                isinstance(vertices, list)
                and 3 <= len(vertices) <= MAX_MESH_VERTICES
                and all(
                    isinstance(vertex, list)
                    and len(vertex) == 3
                    and all(_is_finite_number(component) for component in vertex)
                    for vertex in vertices
                )
            ):
                raise ValueError(
                    f"mesh.vertices must contain 3 to {MAX_MESH_VERTICES} finite 3D points."
                )
            if not (
                isinstance(triangles, list)
                and 1 <= len(triangles) <= MAX_MESH_FACES
                and all(
                    isinstance(face, list)
                    and len(face) == 3
                    and all(isinstance(index, int) and not isinstance(index, bool) for index in face)
                    for face in triangles
                )
            ):
                raise ValueError(
                    f"mesh.triangles must contain 1 to {MAX_MESH_FACES} three-index faces."
                )

            vertex_count = len(vertices)
            for face in triangles:
                if any(index < 0 or index >= vertex_count for index in face):
                    raise ValueError("mesh triangle index is outside the vertex list.")
                if len(set(face)) != 3:
                    raise ValueError("mesh triangles cannot repeat a vertex.")

            points = ",\n".join(
                f"{indent}  [{vertex[0]}, {vertex[1]}, {vertex[2]}]"
                for vertex in vertices
            )
            faces = ",\n".join(
                f"{indent}  [{face[0]}, {face[1]}, {face[2]}]"
                for face in triangles
            )
            return [
                f"{indent}polyhedron(",
                f"{indent}  points = [",
                points,
                f"{indent}  ],",
                f"{indent}  faces = [",
                faces,
                f"{indent}  ]",
                f"{indent});",
            ]

        if kind in {"union", "difference"}:
            children = op.get("shapes")
            if not (
                isinstance(children, list)
                and 1 <= len(children) <= MAX_OPERATIONS
                and all(isinstance(child, dict) for child in children)
            ):
                raise ValueError(f"{kind} requires a non-empty list of shape objects.")
            body: list[str] = []
            for child in children:
                body.extend(render_shape(child, indent + "  ", depth + 1))
            return [f"{indent}{kind}() {{", *body, f"{indent}}}"]

        raise ValueError(f"Unsupported CAD operation: {kind!r}")

    lines = ["$fn = 48;", ""]
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Each CAD operation must be an object.")
        lines.extend(render_shape(operation))
        lines.append("")

    return "\n".join(lines).rstrip()
