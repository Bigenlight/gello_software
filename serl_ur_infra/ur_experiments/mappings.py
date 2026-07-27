"""Task registry for ``--ur-config-module ur_experiments.mappings``.

``run_remote_rlpd_actor.py`` / ``train_rlpd_actor.py`` import upstream's
``experiments.mappings.CONFIG_MAPPING`` and then ``.update()`` it with the
mapping found here, so UR task names live alongside the Franka ones without
patching the vendored tree.
"""

from ur_experiments.cube_in_cup import CubeInCupConfig

CONFIG_MAPPING = {
    "cube_in_cup": CubeInCupConfig,
}
