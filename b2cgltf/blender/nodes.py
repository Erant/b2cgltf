"""The add-on's Geometry Nodes graphs (bpy).

b2c_pose_splat  on the splat: b2ctrain's cage.cu pose_bound from the evaluated cage - each bound splat follows its
                cage triangle (foot point + offset in the triangle frame x the triangle size), turns with it
                (rotation and SH), and grows with it within maxGrowth. The binding is in point attributes (b2c_*).
b2c_offsets     on the cage, after the Armature modifier: adds the positions of a helper mesh (the runtime writes the
                pose correctives and the clip residual there), times a weight.
"""
from __future__ import annotations

import bpy

from .posing import SH_BANDS, sample_dirs

POSE_TREE, OFFSET_TREE = "b2c_pose_splat", "b2c_offsets"


class Graph:
    def __init__(self, name: str):
        self.nt = bpy.data.node_groups.new(name, "GeometryNodeTree")
        self.nt.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        self.nt.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        self.gi, self.go = self.node("NodeGroupInput"), self.node("NodeGroupOutput")

    def node(self, kind: str, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def put(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.link(v, sock)
        else:
            sock.default_value = v

    def attr(self, name: str, dtype: str):
        n = self.node("GeometryNodeInputNamedAttribute", data_type=dtype); n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    def vmath(self, op: str, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        if scale is not None:
            self.put(n.inputs["Scale"], scale)
        return n.outputs["Value" if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else "Vector"]

    def math(self, op: str, a, b=None):
        n = self.node("ShaderNodeMath", operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return n.outputs[0]

    def xyz(self, v):
        n = self.node("ShaderNodeSeparateXYZ"); self.link(v, n.inputs[0])
        return n.outputs

    def rotate_vec(self, v, r):
        n = self.node("FunctionNodeRotateVector"); self.put(n.inputs["Vector"], v); self.link(r, n.inputs["Rotation"])
        return n.outputs[0]

    def rotate_rot(self, rot, by):
        n = self.node("FunctionNodeRotateRotation", rotation_space="GLOBAL")      # by * rot
        self.link(rot, n.inputs["Rotation"]); self.link(by, n.inputs["Rotate By"])
        return n.outputs[0]

    def store(self, geo, name: str, dtype: str, value, sel=None):
        n = self.node("GeometryNodeStoreNamedAttribute", data_type=dtype, domain="POINT")
        self.link(geo, n.inputs["Geometry"]); n.inputs["Name"].default_value = name; self.link(value, n.inputs["Value"])
        if sel is not None:
            self.link(sel, n.inputs["Selection"])
        return n.outputs["Geometry"]


def _sh_basis(g: Graph, l: int, d) -> list:
    """posing.sh_basis as nodes."""
    x, y, z = g.xyz(d)
    m = lambda a, b: g.math("MULTIPLY", a, b)
    sb = lambda a, b: g.math("SUBTRACT", a, b)
    if l == 1:
        c = 0.4886025119029199
        return [m(y, -c), m(z, c), m(x, -c)]
    xx, yy, zz = m(x, x), m(y, y), m(z, z)
    if l == 2:
        c = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
        return [m(m(x, y), c[0]), m(m(y, z), c[1]), m(sb(sb(m(zz, 2.0), xx), yy), c[2]), m(m(x, z), c[3]), m(sb(xx, yy), c[4])]
    c = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
         1.445305721320277, -0.5900435899266435]
    q4 = sb(sb(m(zz, 4.0), xx), yy)
    return [m(m(y, sb(m(xx, 3.0), yy)), c[0]), m(m(m(x, y), z), c[1]), m(m(y, q4), c[2]),
            m(m(z, sb(sb(m(zz, 2.0), m(xx, 3.0)), m(yy, 3.0))), c[3]), m(m(x, q4), c[4]), m(m(z, sb(xx, yy)), c[5]),
            m(m(x, sb(xx, m(yy, 3.0))), c[6])]


def pose_graph(sh_count: int) -> bpy.types.NodeTree:
    """Inputs: Cage (object), Max Growth. Point attributes: b2c_ia/ib/ic (cage vertex indices of the bound triangle),
    b2c_bary (b1, b2, 0), b2c_off, b2c_k0 (canonical triangle size), b2c_r0inv (inverse canonical triangle frame),
    b2c_q0 / b2c_s0 / b2c_sh<k> (the canonical rotation, scale, SH), b2c_bound."""
    g = Graph(POSE_TREE)
    g.nt.interface.new_socket("Cage", in_out="INPUT", socket_type="NodeSocketObject")
    g.nt.interface.new_socket("Max Growth", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.15
    oi = g.node("GeometryNodeObjectInfo", transform_space="RELATIVE"); g.link(g.gi.outputs["Cage"], oi.inputs["Object"])
    cage = oi.outputs["Geometry"]
    pos = g.node("GeometryNodeInputPosition").outputs[0]

    def corner(name):
        s = g.node("GeometryNodeSampleIndex", data_type="FLOAT_VECTOR", domain="POINT")
        g.link(cage, s.inputs["Geometry"]); g.link(pos, s.inputs["Value"]); g.link(g.attr(name, "INT"), s.inputs["Index"])
        return s.outputs[0]
    A, B, Cv = corner("b2c_ia"), corner("b2c_ib"), corner("b2c_ic")
    u, w = g.vmath("SUBTRACT", B, A), g.vmath("SUBTRACT", Cv, A)
    nn = g.vmath("CROSS_PRODUCT", u, w)
    k0, mg = g.attr("b2c_k0", "FLOAT"), g.gi.outputs["Max Growth"]
    k = g.math("SQRT", g.vmath("LENGTH", nn))
    k = g.math("MINIMUM", g.math("MAXIMUM", k, g.math("DIVIDE", k0, mg)), g.math("MULTIPLY", k0, mg))
    ax = g.node("FunctionNodeAxesToRotation", primary_axis="Z", secondary_axis="X")   # columns (e1, n x e1, n)
    g.link(nn, ax.inputs["Primary Axis"]); g.link(u, ax.inputs["Secondary Axis"])
    R = ax.outputs[0]
    bary = g.xyz(g.attr("b2c_bary", "FLOAT_VECTOR"))
    b0 = g.math("SUBTRACT", g.math("SUBTRACT", 1.0, bary[0]), bary[1])
    foot = g.vmath("ADD", g.vmath("ADD", g.vmath("SCALE", A, scale=b0), g.vmath("SCALE", B, scale=bary[0])),
                   g.vmath("SCALE", Cv, scale=bary[1]))
    new_pos = g.vmath("ADD", foot, g.rotate_vec(g.vmath("SCALE", g.attr("b2c_off", "FLOAT_VECTOR"), scale=k), R))
    dq = g.rotate_rot(g.attr("b2c_r0inv", "QUATERNION"), R)                             # R R0^-1
    new_rot = g.rotate_rot(g.attr("b2c_q0", "QUATERNION"), dq)
    new_scale = g.vmath("SCALE", g.attr("b2c_s0", "FLOAT_VECTOR"), scale=g.math("DIVIDE", k, k0))
    bound = g.attr("b2c_bound", "BOOLEAN")

    sp = g.node("GeometryNodeSetPosition")
    g.link(g.gi.outputs["Geometry"], sp.inputs["Geometry"]); g.link(new_pos, sp.inputs["Position"]); g.link(bound, sp.inputs["Selection"])
    geo = g.store(sp.outputs[0], "rotation", "QUATERNION", new_rot, bound)
    geo = g.store(geo, "scale", "FLOAT_VECTOR", new_scale, bound)
    inv = g.node("FunctionNodeInvertRotation"); g.link(dq, inv.inputs[0])
    dqT = inv.outputs[0]
    for l, (a, b) in SH_BANDS.items():            # g(d) = f(dq^T d): values at fixed directions, re-projected
        if b > sh_count:
            break
        D, Yinv = sample_dirs(l)
        coef = [g.attr(f"b2c_sh{i}", "FLOAT_VECTOR") for i in range(a, b)]
        vals = []
        for dk in D:
            acc = None
            for ym, cm in zip(_sh_basis(g, l, g.rotate_vec(tuple(float(x) for x in dk), dqT)), coef):
                t = g.vmath("SCALE", cm, scale=ym); acc = t if acc is None else g.vmath("ADD", acc, t)
            vals.append(acc)
        for mi in range(2 * l + 1):
            acc = None
            for ki in range(2 * l + 1):
                t = g.vmath("SCALE", vals[ki], scale=float(Yinv[mi, ki])); acc = t if acc is None else g.vmath("ADD", acc, t)
            geo = g.store(geo, f"radiance:sh_{a + mi}", "FLOAT_VECTOR", acc, bound)
    g.link(geo, g.go.inputs["Geometry"])
    return g.nt


def offset_graph() -> bpy.types.NodeTree:
    """Inputs: Offsets (a mesh with the cage's vertex count; its positions are the offsets), Weight."""
    g = Graph(OFFSET_TREE)
    g.nt.interface.new_socket("Offsets", in_out="INPUT", socket_type="NodeSocketObject")
    g.nt.interface.new_socket("Weight", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0
    oi = g.node("GeometryNodeObjectInfo", transform_space="ORIGINAL"); g.link(g.gi.outputs["Offsets"], oi.inputs["Object"])
    s = g.node("GeometryNodeSampleIndex", data_type="FLOAT_VECTOR", domain="POINT")
    g.link(oi.outputs["Geometry"], s.inputs["Geometry"]); g.link(g.node("GeometryNodeInputPosition").outputs[0], s.inputs["Value"])
    g.link(g.node("GeometryNodeInputIndex").outputs[0], s.inputs["Index"])
    sp = g.node("GeometryNodeSetPosition"); g.link(g.gi.outputs["Geometry"], sp.inputs["Geometry"])
    g.link(g.vmath("SCALE", s.outputs[0], scale=g.gi.outputs["Weight"]), sp.inputs["Offset"])
    g.link(sp.outputs[0], g.go.inputs["Geometry"])
    return g.nt


def set_input(mod, name: str, value) -> None:
    """Blender 5.3 keeps a node modifier's inputs at mod.properties.inputs.<socket identifier>.value. Setting one
    there does not tag the object for re-evaluation, so tag it."""
    getattr(mod.properties.inputs, mod.node_group.interface.items_tree[name].identifier).value = value
    mod.id_data.update_tag()
