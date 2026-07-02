import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from trimesh import Trimesh
from occwl.compound import Compound
from occwl.solid import Solid

class STEPLoader:
    def __init__(self, filename: str, face_tolerance: float, angle_tolerance: float, max_workers: int):
        self.filename = filename
        self.face_tolerance = face_tolerance
        self.angle_tolerance = angle_tolerance
        self.max_workers = max_workers

    def load(self, solid: Solid):
        vertices, faces = list(), list()
        vertex_offset = 0

        solid.triangulate_all_faces(triangle_face_tol = self.face_tolerance, angle_tol_rads = self.angle_tolerance)
        for face in solid.faces():
            vertex, face = face.get_triangles()
            if len(vertex) == 0 or len(face) == 0:
                continue
            vertices.append(vertex)
            faces.append(face + vertex_offset)
            vertex_offset += len(vertex)

        return Trimesh(vertices = np.vstack(vertices), faces = np.vstack(faces))

    def load_all(self):
        trimeshes = list()
        compound = Compound.load_from_step(self.filename)

        with ThreadPoolExecutor(max_workers = self.max_workers) as executor:
            futures = [executor.submit(self.load, solid) for solid in compound.solids()]
            for future in as_completed(futures):
                trimeshes.append(future.result())

        return trimeshes
