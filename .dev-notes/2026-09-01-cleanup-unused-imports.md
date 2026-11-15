# Cleanup Unused Imports in core-py/domains

## Task
Clean up unused imports in `/home/mana/Documents/Default Project/sloughGPT/packages/core-py/domains/`.

## Steps
1. Run the Python script to find unused imports
2. Process files in batches starting with top offenders:
   - `infrastructure/pugqeep/facade.py` (14 unused)
   - `infrastructure/pugqeep/__init__.py` (8 unused)
   - `collections/scheduler.py` (8 unused)
   - `collections/perception.py` (7 unused)
   - `infrastructure/inference_engine.py` (3 unused)
   - `infrastructure/model_catalog.py` (2 unused)
   - `infrastructure/model_server.py` (3 unused)
   - `learner/data_filter.py` (4 unused)
   - `learner/continual.py` (2 unused)
   - `shell/vm.py` (3 unused)
3. Remove unused imports using the Edit tool
4. Verify with `python3 -m py_compile`
5. Return count of files fixed and imports removed

## Status
- [x] Started
- [x] Completed

## Results
Files fixed: 10
Imports removed: 53

### Files processed:
1. `infrastructure/pugqeep/facade.py` - removed 15 unused imports
2. `infrastructure/pugqeep/__init__.py` - removed 5 unused imports
3. `collections/scheduler.py` - removed 8 unused imports
4. `collections/perception.py` - removed 7 unused imports
5. `infrastructure/inference_engine.py` - removed 4 unused imports
6. `infrastructure/model_catalog.py` - removed 2 unused imports
7. `infrastructure/model_server.py` - removed 3 unused imports
8. `learner/data_filter.py` - removed 3 unused imports
9. `learner/continual.py` - removed 3 unused imports
10. `shell/vm.py` - removed 3 unused imports (top-level only, preserved re-exports)