"""laptop3-side local policy inference for HIL-SERL.

The server keeps training, reward authority, replay and the classifier; this
package moves only the ACTION path onto laptop3 so the blocking inference RPC
leaves the 10 Hz control loop.

Deliberately empty of re-exports: every module here is imported by its full
path (``ur_env.local_policy.runtime``), so importing one of them never drags in
jax for a process that only wanted the finalizer, and vice versa.
"""
