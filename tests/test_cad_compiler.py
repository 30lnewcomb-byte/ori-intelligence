import unittest

from tensorflow_core.cad_compiler import compile_cad_plan


class CadCompilerTests(unittest.TestCase):
    def test_compiles_cube(self):
        source = compile_cad_plan({
            "operations": [{"kind": "cube", "size": [10, 20, 30]}]
        })
        self.assertIn("cube([10, 20, 30], center=true);", source)

    def test_compiles_cylinder(self):
        source = compile_cad_plan({
            "operations": [{"kind": "cylinder", "radius": 4, "height": 12, "center": False}]
        })
        self.assertIn("cylinder(r=4, h=12, center=false);", source)

    def test_compiles_mesh(self):
        source = compile_cad_plan({
            "operations": [{
                "kind": "mesh",
                "vertices": [[0, 0, 0], [10, 0, 0], [0, 10, 0]],
                "triangles": [[0, 1, 2]],
            }]
        })
        self.assertIn("polyhedron(", source)
        self.assertIn("[0, 1, 2]", source)

    def test_rejects_boolean_dimensions(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            compile_cad_plan({
                "operations": [{"kind": "cube", "size": [True, 10, 10]}]
            })

    def test_rejects_non_finite_dimensions(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            compile_cad_plan({
                "operations": [{"kind": "cube", "size": [float("inf"), 10, 10]}]
            })

    def test_rejects_integer_too_large_for_finite_float(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            compile_cad_plan({
                "operations": [{"kind": "cube", "size": [10 ** 10000, 10, 10]}]
            })

    def test_rejects_out_of_range_mesh_index(self):
        with self.assertRaisesRegex(ValueError, "outside the vertex list"):
            compile_cad_plan({
                "operations": [{
                    "kind": "mesh",
                    "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    "triangles": [[0, 1, 3]],
                }]
            })

    def test_rejects_repeated_mesh_vertex(self):
        with self.assertRaisesRegex(ValueError, "cannot repeat"):
            compile_cad_plan({
                "operations": [{
                    "kind": "mesh",
                    "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    "triangles": [[0, 1, 1]],
                }]
            })

    def test_rejects_unknown_operation(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CAD operation"):
            compile_cad_plan({"operations": [{"kind": "sphere", "radius": 10}]})

    def test_compiles_nested_transform(self):
        source = compile_cad_plan({
            "operations": [{
                "kind": "translate",
                "value": [1, 2, 3],
                "shape": {"kind": "cube", "size": [2, 2, 2]},
            }]
        })
        self.assertIn("translate([1, 2, 3])", source)
        self.assertIn("cube([2, 2, 2]", source)

    def test_rejects_deep_nesting(self):
        shape = {"kind": "cube", "size": [1, 1, 1]}
        for _ in range(18):
            shape = {"kind": "translate", "value": [0, 0, 0], "shape": shape}
        with self.assertRaisesRegex(ValueError, "nesting is too deep"):
            compile_cad_plan({"operations": [shape]})

    def test_rejects_excessive_total_operations_across_groups(self):
        union = {
            "kind": "union",
            "shapes": [{"kind": "cube", "size": [1, 1, 1]} for _ in range(128)],
        }
        with self.assertRaisesRegex(ValueError, "more than 128 total operations"):
            compile_cad_plan({"operations": [union]})

    def test_rejects_excessive_total_mesh_vertices(self):
        first_mesh = {
            "kind": "mesh",
            "vertices": [[i, 0, 0] for i in range(2500)],
            "triangles": [[0, 1, 2]],
        }
        second_mesh = {
            "kind": "mesh",
            "vertices": [[i, 1, 0] for i in range(2501)],
            "triangles": [[0, 1, 2]],
        }
        with self.assertRaisesRegex(ValueError, "total mesh vertices"):
            compile_cad_plan({"operations": [first_mesh, second_mesh]})

    def test_rejects_excessive_total_mesh_faces(self):
        faces = [[0, 1, 2] for _ in range(5001)]
        mesh = {
            "kind": "mesh",
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "triangles": faces,
        }
        with self.assertRaisesRegex(ValueError, "total mesh faces"):
            compile_cad_plan({"operations": [mesh, mesh]})

    def test_rejects_excessive_total_operations_across_sibling_roots(self):
        roots = [{"kind": "cube", "size": [1, 1, 1]} for _ in range(128)]
        roots[0] = {
            "kind": "union",
            "shapes": [{"kind": "cube", "size": [1, 1, 1]}],
        }
        with self.assertRaisesRegex(ValueError, "more than 128 total operations"):
            compile_cad_plan({"operations": roots})


if __name__ == "__main__":
    unittest.main()
