"""Project management for CycIF segmentation tool.

A project is a folder containing a project manifest (project.json) plus
subfolders for data, models, exports, and logs.
"""
from .project import CycIFProject, create_project, open_project, is_project_dir

__all__ = ["CycIFProject", "create_project", "open_project", "is_project_dir"]
