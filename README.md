# Assembly Validation
## Code Structure
```
├── config/
    ├── config.yaml : configuration file
├── core/
    ├── action.py : action class
    ├── collision.py : collision detection module using trimesh
    ├── planner.py : path planner module
    ├── state.py : state class
├── data/
    ├── exporter.py : result exporter module using msgpack
    ├── loader.py : step loader and mesh converter using occwl and trimesh
├── test/
├── visualization/
    ├── visualizer.py : result visualizer using PyVista or Open3D
├── main.py
```