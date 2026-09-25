"""Load-time construction of the buoyancy geometry used by cob.CobModel.

Runs once per episode initialisation, never in the simulation loop.
Everything is converted to SI units and flattened into arrays:

- Spheres, ellipsoids, cylinders and capsules keep their analytic sizes.
- Boxes and meshes become closed triangle meshes stored in a BVH whose
  nodes carry precomputed divergence-theorem moments (see cob.pyx).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

# MuJoCo mjtGeom
MJ_SPHERE, MJ_CAPSULE, MJ_ELLIPSOID, MJ_CYLINDER, MJ_BOX, MJ_MESH = (
    2, 3, 4, 5, 6, 7,
)

# Kinds used by cob.pyx (must match the enum in cob.pxd)
SPHERE, ELLIPSOID, CYLINDER, CAPSULE, POLYHEDRON = range(5)

LEAF_SIZE = 4


@dataclass
class CobGeometry:
    """Flattened buoyancy geometry for a set of links"""
    n_links: int
    geom_kind: np.ndarray
    geom_link: np.ndarray
    geom_id: np.ndarray
    geom_root: np.ndarray
    geom_size: np.ndarray
    geom_scale: np.ndarray
    geom_volume: np.ndarray
    geom_centroid: np.ndarray
    geom_rbound: np.ndarray
    bvh: dict = field(default_factory=dict)
    link_volume: np.ndarray = None
    link_overlap: np.ndarray = None


# ---------------------------------------------------------------------------
# Triangle meshes
# ---------------------------------------------------------------------------

def box_triangles(half):
    """Closed, outward oriented box mesh (12 triangles, [12, 3, 3])"""
    hx, hy, hz = half
    verts = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return verts[faces]


def mesh_volume_centroid(tris):
    """Volume and centroid of a closed triangle mesh [n, 3, 3]"""
    det = np.einsum('ij,ij->i', tris[:, 0], np.cross(tris[:, 1], tris[:, 2]))
    volume = det.sum()/6
    centroid = (det[:, None]*tris.sum(axis=1)).sum(axis=0)/(24*volume)
    return volume, centroid


def is_closed(faces):
    """Whether every edge is shared by exactly two consistently oriented
    faces (watertight, manifold, oriented)"""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    directed = {tuple(edge) for edge in edges}
    if len(directed) != len(edges):
        return False
    return all((b, a) in directed for a, b in directed)


def convex_hull_triangles(points):
    """Outward oriented convex hull triangles of a point cloud"""
    from scipy.spatial import ConvexHull  # pylint: disable=import-outside-toplevel
    hull = ConvexHull(points)
    tris = points[hull.simplices]
    center = points.mean(axis=0)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.einsum('ij,ij->i', normals, tris[:, 0] - center) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def mujoco_mesh_triangles(model, geom_i, meters=1.0):
    """Triangles of a mesh geom in its geom frame (SI units)"""
    mesh_i = model.geom_dataid[geom_i]
    vert_adr = model.mesh_vertadr[mesh_i]
    face_adr = model.mesh_faceadr[mesh_i]
    verts = np.array(
        model.mesh_vert[vert_adr:vert_adr+model.mesh_vertnum[mesh_i]],
        dtype=float,
    )/meters
    faces = np.array(
        model.mesh_face[face_adr:face_adr+model.mesh_facenum[mesh_i]],
        dtype=int,
    )
    if not is_closed(faces):
        warnings.warn(
            f'Buoyancy: mesh of geom {geom_i} is not watertight,'
            ' using its convex hull instead'
        )
        return convex_hull_triangles(verts)
    tris = verts[faces]
    volume, _ = mesh_volume_centroid(tris)
    if volume < 0:
        tris = tris[:, [0, 2, 1]]
    return tris


# ---------------------------------------------------------------------------
# BVH with divergence theorem moments
# ---------------------------------------------------------------------------

def triangle_moments(tris):
    """Per triangle moments [n, 16]: det, N, det*s, s N^T (see cob.pyx)"""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    det = np.einsum('ij,ij->i', a, np.cross(b, c))
    normal = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
    total = a + b + c
    return np.concatenate([
        det[:, None],
        normal,
        det[:, None]*total,
        np.einsum('ni,nj->nij', total, normal).reshape(-1, 9),
    ], axis=1)


def build_bvh(tris, node_offset=0, tri_offset=0):
    """Build a BVH over triangles [n, 3, 3].

    Returns the node arrays (indices already offset for concatenation)
    and the triangles reordered so that leaves are contiguous.
    """
    moments = triangle_moments(tris)
    centroids = tris.mean(axis=1)
    order = np.arange(len(tris))
    nodes = []  # [center, half, moments, left, right, start, count]

    def build(start, end):
        node_i = len(nodes)
        nodes.append(None)
        idx = order[start:end]
        points = tris[idx].reshape(-1, 3)
        lo, hi = points.min(axis=0), points.max(axis=0)
        node = [
            0.5*(lo + hi), 0.5*(hi - lo), moments[idx].sum(axis=0),
            -1, -1, tri_offset + start, end - start,
        ]
        if end - start > LEAF_SIZE:
            axis = np.argmax(np.ptp(centroids[idx], axis=0))
            sort = np.argsort(centroids[idx, axis], kind='stable')
            order[start:end] = idx[sort]
            mid = (start + end)//2
            node[3] = node_offset + build(start, mid)
            node[4] = node_offset + build(mid, end)
        nodes[node_i] = node
        return node_i

    build(0, len(tris))
    return {
        'center': np.array([node[0] for node in nodes]),
        'half': np.array([node[1] for node in nodes]),
        'moments': np.array([node[2] for node in nodes]),
        'left': np.array([node[3] for node in nodes], dtype=np.intc),
        'right': np.array([node[4] for node in nodes], dtype=np.intc),
        'tri_start': np.array([node[5] for node in nodes], dtype=np.intc),
        'tri_count': np.array([node[6] for node in nodes], dtype=np.intc),
        'tris': tris[order].reshape(-1, 9),
    }


def empty_bvh():
    """Empty BVH arrays"""
    return {
        'center': np.zeros([0, 3]),
        'half': np.zeros([0, 3]),
        'moments': np.zeros([0, 16]),
        'left': np.zeros(0, dtype=np.intc),
        'right': np.zeros(0, dtype=np.intc),
        'tri_start': np.zeros(0, dtype=np.intc),
        'tri_count': np.zeros(0, dtype=np.intc),
        'tris': np.zeros([0, 9]),
    }


def concatenate_bvhs(bvhs):
    """Concatenate BVHs built with consistent offsets"""
    if not bvhs:
        return empty_bvh()
    return {key: np.concatenate([bvh[key] for bvh in bvhs]) for key in bvhs[0]}


# ---------------------------------------------------------------------------
# Primitive properties and point membership
# ---------------------------------------------------------------------------

def primitive_volume(kind, size):
    """Exact volume of an analytic primitive"""
    if kind == SPHERE:
        return 4/3*np.pi*size[0]**3
    if kind == ELLIPSOID:
        return 4/3*np.pi*size[0]*size[1]*size[2]
    if kind == CYLINDER:
        return np.pi*size[0]**2*2*size[1]
    if kind == CAPSULE:
        return np.pi*size[0]**2*(2*size[1] + 4/3*size[0])
    raise ValueError(kind)


def primitive_rbound(kind, size):
    """Bounding radius about the primitive centre"""
    if kind == SPHERE:
        return size[0]
    if kind == ELLIPSOID:
        return max(size[:3])
    if kind == CYLINDER:
        return float(np.hypot(size[0], size[1]))
    if kind == CAPSULE:
        return size[0] + size[1]
    raise ValueError(kind)


def points_inside(kind, size, points, tris=None):
    """Boolean mask of points (geom frame) inside a geom"""
    x, y, z = points.T
    if kind == SPHERE:
        return x**2 + y**2 + z**2 <= size[0]**2
    if kind == ELLIPSOID:
        return (x/size[0])**2 + (y/size[1])**2 + (z/size[2])**2 <= 1
    if kind == CYLINDER:
        return (x**2 + y**2 <= size[0]**2) & (np.abs(z) <= size[1])
    if kind == CAPSULE:
        zc = np.clip(z, -size[1], size[1])
        return x**2 + y**2 + (z - zc)**2 <= size[0]**2
    return winding_number(tris, points) > 0.5


def winding_number(tris, points, chunk=4096):
    """Generalised winding number of points w.r.t. a closed mesh"""
    result = np.zeros(len(points))
    for start in range(0, len(points), chunk):
        pts = points[start:start+chunk]
        a = tris[None, :, 0] - pts[:, None]
        b = tris[None, :, 1] - pts[:, None]
        c = tris[None, :, 2] - pts[:, None]
        la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
        det = np.einsum('pti,pti->pt', a, np.cross(b, c))
        den = (
            la*lb*lc
            + np.einsum('pti,pti->pt', a, b)*lc
            + np.einsum('pti,pti->pt', b, c)*la
            + np.einsum('pti,pti->pt', c, a)*lb
        )
        result[start:start+chunk] = np.arctan2(det, den).sum(axis=1)/(2*np.pi)
    return result


def quat2mat(quat):
    """Rotation matrix from MuJoCo (w, x, y, z) quaternion"""
    w, x, y, z = quat
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------

@dataclass
class GeomInfo:
    """Buoyant geom description in SI units"""
    geom_id: int
    link: int
    kind: int
    size: np.ndarray
    pos: np.ndarray          # In body frame
    rot: np.ndarray          # In body frame
    tris: np.ndarray = None  # Geom frame, polyhedra only


def link_geoms(model, body_ids, geom_group=2, meters=1.0):
    """Buoyant geoms of each body (list of GeomInfo)"""
    geoms = []
    for link_i, body_id in enumerate(body_ids):
        for geom_i in np.flatnonzero(model.geom_bodyid == body_id):
            if model.geom_group[geom_i] != geom_group:
                continue
            mj_type = int(model.geom_type[geom_i])
            size = np.array(model.geom_size[geom_i], dtype=float)/meters
            tris = None
            if mj_type == MJ_SPHERE:
                kind, size = SPHERE, size[:1]
            elif mj_type == MJ_ELLIPSOID:
                kind = ELLIPSOID
            elif mj_type == MJ_CYLINDER:
                kind, size = CYLINDER, size[:2]
            elif mj_type == MJ_CAPSULE:
                kind, size = CAPSULE, size[:2]
            elif mj_type == MJ_BOX:
                kind, tris = POLYHEDRON, box_triangles(size)
            elif mj_type == MJ_MESH:
                kind = POLYHEDRON
                tris = mujoco_mesh_triangles(model, geom_i, meters)
            else:
                warnings.warn(
                    f'Buoyancy: geom {geom_i} of type {mj_type} is not'
                    ' supported and is ignored'
                )
                continue
            geoms.append(GeomInfo(
                geom_id=int(geom_i),
                link=link_i,
                kind=kind,
                size=np.pad(size, (0, 3-len(size))),
                pos=np.array(model.geom_pos[geom_i], dtype=float)/meters,
                rot=quat2mat(model.geom_quat[geom_i]),
                tris=tris,
            ))
    return geoms


def geom_bounds(geom):
    """Axis aligned bounds of a geom in its body frame"""
    if geom.kind == POLYHEDRON:
        points = geom.tris.reshape(-1, 3)
    else:
        radius = primitive_rbound(geom.kind, geom.size)
        points = np.array(np.meshgrid(*[[-radius, radius]]*3)).reshape(3, -1).T
    points = points @ geom.rot.T + geom.pos
    return points.min(axis=0), points.max(axis=0)


def estimate_overlap(geoms, n_samples=20000, seed=0):
    """Monte Carlo estimate of union volume / sum of volumes for a link"""
    if len(geoms) < 2:
        return 1.0
    bounds = np.array([geom_bounds(geom) for geom in geoms])
    lo, hi = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    rng = np.random.default_rng(seed)
    points = lo + (hi - lo)*rng.random([n_samples, 3])
    count = np.zeros(n_samples, dtype=int)
    for geom in geoms:
        local = (points - geom.pos) @ geom.rot
        count += points_inside(geom.kind, geom.size, local, geom.tris)
    total = count.sum()
    return float(np.count_nonzero(count)/total) if total else 1.0


def build_cob_geometry(model, body_ids, geom_group=2, meters=1.0, overlap='ignore'):
    """Build the flattened CoB geometry for the given MuJoCo bodies"""
    return assemble_cob_geometry(
        geoms=link_geoms(model, body_ids, geom_group=geom_group, meters=meters),
        n_links=len(body_ids),
        overlap=overlap,
    )


def assemble_cob_geometry(geoms, n_links, overlap='ignore'):
    """Flatten a list of GeomInfo into a CobGeometry.

    overlap='scale' rescales each link's geoms by (union volume / sum of
    volumes) so overlapping geoms are not counted twice when the link is
    fully submerged. 'ignore' keeps the plain sum.
    """
    kinds, links, ids, roots, sizes = [], [], [], [], []
    volumes, centroids, rbounds, bvhs = [], [], [], []
    n_nodes = n_tris = 0
    for geom in geoms:
        root = -1
        if geom.kind == POLYHEDRON:
            volume, centroid = mesh_volume_centroid(geom.tris)
            rbound = float(np.max(np.linalg.norm(
                geom.tris.reshape(-1, 3) - centroid, axis=1,
            )))
            bvh = build_bvh(geom.tris, node_offset=n_nodes, tri_offset=n_tris)
            root = n_nodes
            n_nodes += len(bvh['left'])
            n_tris += len(geom.tris)
            bvhs.append(bvh)
        else:
            volume = primitive_volume(geom.kind, geom.size)
            centroid = np.zeros(3)
            rbound = primitive_rbound(geom.kind, geom.size)
        kinds.append(geom.kind)
        links.append(geom.link)
        ids.append(geom.geom_id)
        roots.append(root)
        sizes.append(geom.size)
        volumes.append(volume)
        centroids.append(centroid)
        rbounds.append(rbound)

    links = np.array(links, dtype=np.intc)
    volumes = np.array(volumes, dtype=float)
    link_overlap = np.ones(n_links)
    for link_i in range(n_links):
        link_geoms_i = [geom for geom in geoms if geom.link == link_i]
        if len(link_geoms_i) > 1:
            link_overlap[link_i] = estimate_overlap(link_geoms_i)
    scale = np.ones(len(geoms))
    if overlap == 'scale':
        scale = link_overlap[links] if len(geoms) else scale
    elif overlap != 'ignore':
        raise ValueError(f'Unknown overlap mode {overlap!r}')
    link_volume = np.array([
        np.sum(volumes[links == link_i]*scale[links == link_i])
        for link_i in range(n_links)
    ])
    return CobGeometry(
        n_links=n_links,
        geom_kind=np.array(kinds, dtype=np.intc),
        geom_link=links,
        geom_id=np.array(ids, dtype=np.intc),
        geom_root=np.array(roots, dtype=np.intc),
        geom_size=np.array(sizes, dtype=float).reshape(-1, 3),
        geom_scale=scale,
        geom_volume=volumes,
        geom_centroid=np.array(centroids, dtype=float).reshape(-1, 3),
        geom_rbound=np.array(rbounds, dtype=float),
        bvh=concatenate_bvhs(bvhs),
        link_volume=link_volume,
        link_overlap=link_overlap,
    )
