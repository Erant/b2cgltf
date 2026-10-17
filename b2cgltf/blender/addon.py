"""The add-on's Blender UI (bpy): File > Import entries, the 3D view sidebar's "b2c" panel, the preferences."""
from __future__ import annotations

import bpy
from bpy.props import CollectionProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import importer, runtime


def _refresh(_self, _ctx):
    runtime.reset()


class B2C_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__   # replaced by register() with the add-on's package

    correctives_path: StringProperty(
        name="Corrective weights", subtype="FILE_PATH", update=_refresh,
        description="MHR's pose-corrective weights (.npz), exported with "
                    "`python -m b2cgltf.blender.correctives OUT.npz` where torch and the MHR model are installed")

    def draw(self, _ctx):
        self.layout.prop(self, "correctives_path")


def _prefs_path(ctx) -> str:
    a = ctx.preferences.addons.get(runtime.PREFS["package"] or "")
    return a.preferences.correctives_path if a else ""


class B2C_OT_import_subject(bpy.types.Operator, ImportHelper):
    """Import a b2c subject file: a rigged one as armature, cage and posable Gaussian splat; an unrigged one as the
    static splat"""
    bl_idname = "import_scene.b2c_subject"
    bl_label = "Import b2c Subject"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".glb"
    filter_glob: StringProperty(default="*.glb", options={"HIDDEN"})

    def execute(self, ctx):
        if self.filepath.endswith(".clip.glb"):
            self.report({"ERROR"}, "that is a clip file: import the subject first, then File > Import > b2c Clip")
            return {"CANCELLED"}
        try:
            r = importer.import_subject(ctx, self.filepath, correctives=None)
        except Exception as e:   # noqa: BLE001 - shown to the user
            self.report({"ERROR"}, f"b2c: {e}")
            return {"CANCELLED"}
        rep = r["report"]
        if r["cage"] is None:
            self.report({"WARNING"}, f"{rep['subject']}: not rigged - imported the static splat ({rep['splats']} splats)")
        else:
            for o in ctx.view_layer.objects.selected:
                o.select_set(False)
            r["armature"].select_set(True); ctx.view_layer.objects.active = r["armature"]
            self.report({"INFO"}, f"{rep['subject']}: {rep['splats']} splats, {rep['joints']} joints; "
                                  f"correctives {rep['correctives']}")
        return {"FINISHED"}


class B2C_OT_import_clip(bpy.types.Operator, ImportHelper):
    """Import b2c clip files onto the selected b2c rig, as actions (the last one becomes active)"""
    bl_idname = "import_scene.b2c_clip"
    bl_label = "Import b2c Clip"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".glb"
    filter_glob: StringProperty(default="*.clip.glb", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, ctx):
        return importer.find_rig(ctx.active_object) is not None

    def execute(self, ctx):
        import os
        cage = importer.find_rig(ctx.active_object)
        paths = [os.path.join(self.directory, f.name) for f in self.files if f.name] or [self.filepath]
        for p in paths:
            try:
                r = importer.import_clip(ctx, cage, p)
            except Exception as e:   # noqa: BLE001
                self.report({"ERROR"}, f"b2c: {os.path.basename(p)}: {e}")
                return {"CANCELLED"}
            self.report({"INFO"}, f"{r['clip']}: {r['frames']} frames at {r['fps']:g} fps"
                                  f"{'' if r['residual'] else ' (no residual)'}")
        return {"FINISHED"}


class B2C_OT_rest_pose(bpy.types.Operator):
    """Unassign the clip and clear the pose: the rig at its bind pose, ready to pose by hand"""
    bl_idname = "b2c.rest_pose"
    bl_label = "Clear Clip and Pose"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, ctx):
        return importer.find_rig(ctx.active_object) is not None

    def execute(self, ctx):
        arm = importer.find_rig(ctx.active_object)["b2c_armature"]
        if arm.animation_data:
            arm.animation_data.action = None
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0); pb.rotation_euler = (0, 0, 0); pb.scale = (1, 1, 1)
        return {"FINISHED"}


class B2C_PT_rig(bpy.types.Panel):
    bl_label = "b2c Rig"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "b2c"

    @classmethod
    def poll(cls, ctx):
        return importer.find_rig(ctx.active_object) is not None

    def draw(self, ctx):
        cage = importer.find_rig(ctx.active_object)
        arm = cage["b2c_armature"]
        lay = self.layout
        info = cage["b2c"]
        lay.label(text=f"Subject {info['subject_id'][:8]}")
        act = arm.animation_data.action if arm.animation_data else None
        lay.label(text=f"Clip: {act.name if act else '(none)'}"
                       + ("" if not act or runtime.clip_path(act) else "  (clip file not found)"))
        row = lay.row(align=True)
        row.operator(B2C_OT_import_clip.bl_idname, text="Import Clip", icon="IMPORT")
        row.operator(B2C_OT_rest_pose.bl_idname, text="Clear", icon="ARMATURE_DATA")
        col = lay.column(align=True)
        col.prop(cage, '["b2c_residual_weight"]', text="Clip residual", slider=True)
        status = runtime.corrective_status(cage)
        sub = col.column(); sub.enabled = status == "on"
        sub.prop(cage, '["b2c_corrective_weight"]', text="Pose correctives", slider=True)
        if status != "on":
            lay.label(text=f"Correctives {status}", icon="INFO")


def _menu(self, _ctx):
    self.layout.operator(B2C_OT_import_subject.bl_idname, text="b2c Subject (.glb)")
    self.layout.operator(B2C_OT_import_clip.bl_idname, text="b2c Clip (.clip.glb)")


CLASSES = (B2C_Preferences, B2C_OT_import_subject, B2C_OT_import_clip, B2C_OT_rest_pose, B2C_PT_rig)


def register(package: str | None = None):
    runtime.PREFS["package"] = package
    B2C_Preferences.bl_idname = package or __package__
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(_menu)
    runtime.register()


def unregister():
    runtime.unregister()
    bpy.types.TOPBAR_MT_file_import.remove(_menu)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
