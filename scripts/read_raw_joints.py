"""Read raw GELLO servo angles (no offset, no pi/2 snapping) for fine calibration.

Read-only: constructs DynamixelDriver (which disables torque on init) and only calls
get_joints(). Never enables torque. Prints each servo's angle in radians and degrees so
a fractional joint_offset can be derived directly (offset_i = raw_i for sign=+1, start=0).
"""
import numpy as np
import tyro
from gello.dynamixel.driver import DynamixelDriver


def main(port: str = "/dev/ttyUSB0", num_servos: int = 8, samples: int = 20) -> None:
    ids = list(range(1, num_servos + 1))
    driver = DynamixelDriver(ids, port=port, baudrate=57600)
    for _ in range(10):
        driver.get_joints()  # warmup
    acc = np.zeros(num_servos)
    for _ in range(samples):
        acc += driver.get_joints()
    raw = acc / samples
    print("\nraw servo angles (id : rad : deg):")
    for i, dxl_id in enumerate(ids):
        print(f"  ID {dxl_id}:  {raw[i]:+.4f} rad   {np.rad2deg(raw[i]):+.2f} deg")
    print("\njoint_offsets candidate (= raw, so sim reads 0 at this held pose):")
    print("  [" + ", ".join(f"{v:.4f}" for v in raw[:7]) + "]   # J1..J7")
    driver.close()


if __name__ == "__main__":
    tyro.cli(main)
