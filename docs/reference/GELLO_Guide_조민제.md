# GELLO Guide

> 📎 **이 문서는 동료(조민제)가 작성한 외부 세팅 가이드를 참고용으로 가져온 것입니다.** 본문은 원본 그대로 보존하며, 아래 내용은 저자 개인 환경(도커 이미지·포트·경로 등)을 전제로 합니다.
> 현재 RWH 워크플로 기준의 세팅·실행 방법은 저장소의 [README](../../README.md)와 [docs/sim](../sim), [docs/ros2](../ros2)를 우선 따르세요. 이 가이드는 배경 참고 자료로만 활용하시면 됩니다.

- Gello Official
    
    https://wuphilipp.github.io/gello_site/
    
    [https://github.com/wuphilipp/gello_software](https://github.com/wuphilipp/gello_software)
    
    [https://github.com/wuphilipp/gello_mechanical](https://github.com/wuphilipp/gello_mechanical)
    
- **Franka Panda**
    - Franka용 추가 세팅
        - Ros2설치 `/gello/ros2`
            
            ```bash
            apt-get update
            apt-get install -y locales curl gnupg lsb-release software-properties-common
            
            locale-gen en_US en_US.UTF-8
            update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
            
            add-apt-repository universe -y
            
            curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
              -o /usr/share/keyrings/ros-archive-keyring.gpg
            
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
              > /etc/apt/sources.list.d/ros2.list
            
            apt-get update
            apt-get install -y \
              ros-humble-ros-base \
              python3-colcon-common-extensions \
              python3-colcon-mixin \
              python3-vcstool \
              python3-rosdep \
              ros-humble-rmw-cyclonedds-cpp \
              ros-humble-pinocchio \
              build-essential \
              cmake \
              libeigen3-dev \
              libfmt-dev \
              libpoco-dev
              
              
              
            rosdep init || true
            rosdep update
            
            echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
            source /opt/ros/humble/setup.bash
            
            which ros2
            ros2 --help
            ```
            
        - libfranka 0.9.2
            
            ```bash
            cd /tmp
            
            apt-get update
            apt-get install -y build-essential cmake git libpoco-dev libeigen3-dev libfmt-dev
            
            git clone --recursive https://github.com/frankaemika/libfranka
            cd libfranka
            
            git checkout 0.9.2
            git submodule update --init --recursive
            
            mkdir -p build
            cd build
            
            cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
            cmake --build . -j$(nproc)
            
            # /tmp/libfranka/build
            cpack -G DEB
            ls -lh *.deb
            dpkg -i ./*.deb
            
            # backup deb
            mkdir -p /gello/artifacts
            cp /tmp/libfranka/build/libfranka-0.9.2-x86_64.deb /gello/artifacts/
            ls -lh /gello/artifacts/
            
            ldconfig
            ldconfig -p | grep franka
            
            # test
            cd /tmp/libfranka/build/examples
            ls
            
            ./communication_test 172.16.0.2
            ```
            
        - mulitpanda_ros2
            
            ```bash
            mkdir -p /workspace/panda_ros2_ws/src
            cd /workspace/panda_ros2_ws/src
            git clone --recursive https://github.com/tenfoldpaper/multipanda_ros2.git
            
            cd /workspace/panda_ros2_ws
            source /opt/ros/humble/setup.bash
            rosdep install --from-paths src --ignore-src -r -y
            colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
            ```
            
- **UR5**
- Piper용 STL ← 나중에 필요하신 분 프린트해서 사용하세요!
    
    [flanger.zip](GELLO%20Guide/flanger.zip)
    

---

![KakaoTalk_20260618_121537745.jpg](GELLO%20Guide/85bb5bff-af7b-41b7-88f8-a8fb9b7664d5.png)

다음과 같이 파워선, TTL 선, C타입 선(→컴퓨터)을 연결하면 됨 (극성주의)

### Dynamixel Wizard 설치 (motor ID 세팅, 이미 세팅된 기체에는 불필요)

처음 gello를 연결하게되면, 모든 joint의 모터 ID가 같은 ID로 잡혀서 다른 ID로 분리해주는 작업을 해야함. 

[https://emanual.robotis.com/docs/kr/software/dynamixel/dynamixel_wizard2/](https://emanual.robotis.com/docs/kr/software/dynamixel/dynamixel_wizard2/) 해당 링크를 통해 각 os에 맞게 설치한다.

Scan option: Protocol 2.0, tty<연결된곳>, 57600bps에서 scan

- ID Inspection ([Official)](https://emanual.robotis.com/docs/kr/software/dynamixel/dynamixel_wizard2/) ⇒ 여러 모터 id를 동시에 바꿀수 있게 해줌
    
    `상단바 Tools → ID inspection`
    
    ![image.png](GELLO%20Guide/image.png)
    
    ![각 모터에 할당된 ID를 다른 번호로 바꿔주면 됨](GELLO%20Guide/image%201.png)
    
    각 모터에 할당된 ID를 다른 번호로 바꿔주면 됨
    
    ![다시 scan하면 잘 잡히는걸 볼 수 있음.](GELLO%20Guide/image%202.png)
    
    다시 scan하면 잘 잡히는걸 볼 수 있음.
    
- Franka (Motor id 세팅완료)
    
    
    ![image.png](GELLO%20Guide/image%203.png)
    
    - Gripper ID: 005
    - Joint 1 ID: 001
    - Joint 2 ID: 007
    - Joint 3 ID: 004
    - Joint 4 ID: 008
    - Joint 5 ID: 006
    - Joint 6 ID: 002
    - Joint 7 ID: 003
    

### Docker (Panda까지 세팅완료)

```bash
# pull
docker pull minje227/gello:RWH_Gello_PandaV1.0

docker exec -it <container> bash

xhost +local:root
  docker run -it \
    --name gello_panda_saved \
    --privileged \
    --net=host \
    -v /dev:/dev \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    minje227/gello:RWH_Gello_PandaV1.0 bash
    
  
```

`docker pull minje227/gello:RWH_Gello_PandaV1.0` ← 해당 docker image기준, sim&Real Panda 세팅 완료!

**업데이트하시면 해당 docker image에 commit해서 최신화 해주시면 감사하겠습니다!**

- **Offeset Calibration하기 (Gello 첫 세팅 시에만!)**
    
    Calibration은 처음 사용하는 로봇팔이나 세팅이 바꼈을 경우 다시 해야하며 아래 사진과 같은 자세를 유지하며 gello_get_offset.py를 실행하면 된다.
    
    ![image.png](GELLO%20Guide/image%204.png)
    
    ```bash
    # Franka Offset Calibration
    python3 scripts/gello_get_offset.py \
        --start-joints 0 0 0 -1.57 0 1.57 0 \
        --joint-signs 1 -1 1 -1 1 -1 1 \
        --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0
    
    # UR Offset Calibration   
    python scripts/gello_get_offset.py \
        --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \
        --joint-signs 1 1 -1 1 1 1 \
        --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6
    ```
    

## Sim 테스트 (+Offset Calibration)

### Panda Sim `(calibration 완료)`

- **MuJoCo***→Calibration 테스트를 하고, 잘 움직이는지 확인할 때 사용*
    
    ```bash
    python3 experiments/launch_yaml.py --left-config-path configs/rwh_panda.yaml
    ```
    
    [Screencast from 2026-06-17 18-46-11.webm](GELLO%20Guide/Screencast_from_2026-06-17_18-46-11.webm)
    
- **Rviz** *→  Real robot연결 직전에 의도한대로 움직이는지 확인하기 위해 사용. (중요!)*
    
    ```r
    docker exec -it rwh_gello_impl bash
    
    source /opt/ros/humble/setup.bash
    source /workspace/panda_ros2_ws/install/setup.bash
    source /gello/ros2/install/setup.bash
    
    ros2 launch franka_bringup rwh_gello_rviz.launch.py gello_com_port:=/dev/ttyUSB0
    ```
    
    ```r
    ros2 topic echo /gello/joint_states --once
    ros2 topic echo /joint_states --once
    ```
    

### UR Sim

- **Offeset Calibration하기 (첫 세팅 시에만!)**
    
    Calibration은 처음 사용하는 로봇팔이나 세팅이 바꼈을 경우 다시 해야하며 아래 사진과 같은 자세를 유지하며 gello_get_offset.py를 실행하면 된다.
    
    ![image.png](GELLO%20Guide/image%204.png)
    
    ```bash
    docker exec -it gello_panda_dev5 bash
    
    # Offset Calibration
    python3 scripts/gello_get_offset.py \
        --start-joints 0 0 0 -1.57 0 1.57 0 \
        --joint-signs 1 -1 1 -1 1 -1 1 \
        --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0
    ```
    

**MuJoCo 시뮬레이터 + Gello 연결** 

```bash
python3 experiments/launch_yaml.py --left-config-path configs/rwh_ur.yaml
```

## Real Robot 테스트

### Panda (세팅완료)

`주의! gello에 찍힌 점들이 한쪽면에 정렬되어야 정상작동하도록 calibratin해둠!`

![image.png](GELLO%20Guide/image%205.png)

[20260618_110333.mp4](GELLO%20Guide/20260618_110333.mp4)

- **Real Panda + Gello 연결**
    
    `⇒ 실시간성이 부족하다면, Joint impedance controller의 P,D gain을 조정`
    
    ```bash
    source /opt/ros/humble/setup.bash
    source /workspace/panda_ros2_ws/install/setup.bash
    source /gello/ros2/install/setup.bash
    
    ROBOT_IP=172.29.0.2 \
    USE_RVIZ=true \
    GELLO_COM_PORT=/dev/ttyUSB0 \
    /workspace/panda_ros2_ws/rwh_gello_realtime_dashboard.sh
    ```
    
    ```r
    # 아래 명령으로 확인
    ros2 topic echo /gello/joint_states --once
    ros2 control list_controllers
    ros2 action list | grep panda_gripper
    ```
    
    - ~~한개씩~~~
        
        ```bash
        # 1. Panda arm + hand action server
        docker exec -it rwh_gello_impl bash
        cd /workspace/panda_ros2_ws
        source /opt/ros/humble/setup.bash
        source install/setup.bash
        ros2 launch franka_bringup rwh_franka.launch.py robot_ip:=172.29.0.2 use_rviz:=true
        
        # 2. GELLO publisher
        docker exec -it rwh_gello_impl bash
        cd /gello/ros2
        source /opt/ros/humble/setup.bash
        source install/setup.bash
        ros2 launch franka_gello_state_publisher main.launch.py config_file:=rwh_panda_ros2.yaml com_port:=/dev/ttyUSB0
        
        # 3. gripper manager
        docker exec -it rwh_gello_impl bash
        cd /gello/ros2
        source /opt/ros/humble/setup.bash
        source install/setup.bash
        ros2 launch franka_gripper_manager rwh_franka_gripper_client.launch.py
        ```
        

### UR

**Real UR + Gello 연결** 

```bash
python3 experiments/launch_yaml.py --left-config-path configs/rwh_ur.yaml

# UR sim
xvfb-run -a python3 experiments/launch_nodes.py --robot sim_ur --robot_port 6001

# GELLO controller
python experiments/run_env.py --agent gello --robot_port 6001 --gello_port /dev/ttyUSB0

# Real UR
python experiments/launch_nodes.py --robot ur --robot_ip <UR_IP> --robot_port 6001
```

---

- …
    
    [GELLO 구매정리](https://app.notion.com/p/GELLO-32263918d42a80a9ae35d14ef1161893?pvs=21)
    
    [https://docs.google.com/document/d/1bdvqPiW4jI5PJm6aIFizhcTuyVJZGEIgqbJe92IoyaM/edit?tab=t.0#heading=h.hbbn0pp1i7p0](https://docs.google.com/document/d/1bdvqPiW4jI5PJm6aIFizhcTuyVJZGEIgqbJe92IoyaM/edit?tab=t.0#heading=h.hbbn0pp1i7p0)
    
    [연구실 노트북 현황](https://app.notion.com/p/Laptop-Configuration-5fb0f4654682428aa137106c1a4206e6?pvs=21)