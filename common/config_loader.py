"""
共享配置加载器 —— Spring Boot 风格 profile 切换

所有模块统一通过此模块获取配置，切换环境只需修改 ``config-dev.yaml`` 中的 ``profile`` 值。

文件结构::

    common/
        config-dev.yaml          ← 启动配置，仅含 profile: main|test
        config-dev-main.yaml     ← 生产环境配置
        config-dev-test.yaml     ← 测试环境配置

用法::

    from common.config_loader import get_config_path, load_config

    config = load_config()           # 返回当前 profile 的完整配置 dict
    mqtt_cfg = config['mqtt']

    path = get_config_path()         # 返回当前 profile 配置文件的路径
    redis = RedisHelper(path)        # 传给需要文件路径的组件

环境变量覆盖:
    ``UAV_PROFILE=test`` 可临时覆盖配置文件中的 profile 值（用于容器/CI 场景）。
"""

import os
import sys
import yaml

# 配置目录基础路径
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 启动配置文件（bootstrap）
_BOOTSTRAP_PATH = os.path.join(_BASE_DIR, "config-dev.yaml")

# 缓存
_profile: str = None
_config_path: str = None
_config: dict = None


def _detect_profile() -> str:
    """检测当前激活的 profile。

    优先级：环境变量 UAV_PROFILE > bootstrap 文件 profile 字段 > 默认 'main'
    """
    env_profile = os.environ.get("UAV_PROFILE", "").strip()
    if env_profile:
        return env_profile

    if os.path.exists(_BOOTSTRAP_PATH):
        try:
            with open(_BOOTSTRAP_PATH, "r", encoding="utf-8") as f:
                bootstrap = yaml.safe_load(f) or {}
            return bootstrap.get("profile", "main")
        except Exception:
            pass

    return "main"


def get_config_path() -> str:
    """返回当前 profile 对应的配置文件路径。

    Returns:
        str: 如 ``/tensorrt/.../common/config-dev-main.yaml``
    """
    global _profile, _config_path
    if _config_path is not None:
        return _config_path

    _profile = _detect_profile()
    _config_path = os.path.join(_BASE_DIR, f"config-dev-{_profile}.yaml")
    return _config_path


def load_config() -> dict:
    """加载当前 profile 的完整配置。

    Returns:
        dict: 解析后的 YAML 配置；文件缺失或解析失败时进程退出。
    """
    global _config
    if _config is not None:
        return _config

    path = get_config_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Error: 配置文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: 配置文件加载失败 {path}: {e}", file=sys.stderr)
        sys.exit(1)

    return _config


def get_profile() -> str:
    """返回当前激活的 profile 名称。"""
    global _profile
    if _profile is None:
        get_config_path()
    return _profile


def reset_cache():
    """重置缓存（仅测试用）。"""
    global _profile, _config_path, _config
    _profile = None
    _config_path = None
    _config = None