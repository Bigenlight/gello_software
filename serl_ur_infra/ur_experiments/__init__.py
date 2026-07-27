"""UR7e HIL-SERL task configurations.

Named ``ur_experiments`` rather than ``experiments`` on purpose: the actor
entry points put ``third_party/hil-serl/examples`` on ``sys.path`` ahead of
``serl_ur_infra``, so a package called ``experiments`` here would be shadowed
by upstream's and ``--ur-config-module`` would silently load the wrong one.
"""
