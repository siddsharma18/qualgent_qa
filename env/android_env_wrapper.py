from android_env import loader
from android_env.components import config_classes
import os

def make_env(task_path=None):
    """
    Create an AndroidEnv instance with proper configuration.
    
    Args:
        task_path: Path to a task textproto file. If None, uses a default simple task.
    
    Returns:
        tuple: (env, obs) where env is the AndroidEnv instance and obs is the initial observation
    
    Requirements:
        - Android SDK installed and configured
        - Android Virtual Device (AVD) created
        - ADB (Android Debug Bridge) available
        - Emulator available
    
    Example setup:
        1. Install Android Studio
        2. Create an AVD (e.g., Pixel_2_API_30)
        3. Set ANDROID_SDK_ROOT environment variable
        4. Ensure adb and emulator are in PATH
    """
    # Use default task path if none provided
    if task_path is None:
        task_path = os.path.join(os.path.dirname(__file__), 'simple_task.textproto')
    
    # Check if task file exists
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Task file not found: {task_path}")
    
    # Create a configuration with the task file
    config = config_classes.AndroidEnvConfig(
        task=config_classes.FilesystemTaskConfig(path=task_path),
        simulator=config_classes.EmulatorConfig(
            emulator_launcher=config_classes.EmulatorLauncherConfig(
                emulator_path=os.path.expanduser('~/Android/Sdk/emulator/emulator'),
                android_sdk_root=os.path.expanduser('~/Android/Sdk'),
                android_avd_home=os.path.expanduser('~/.android/avd'),
                avd_name='Pixel_2_API_30',  # Default AVD name
                run_headless=False,
            ),
            adb_controller=config_classes.AdbControllerConfig(
                adb_path=os.path.expanduser('~/Android/Sdk/platform-tools/adb')
            ),
        ),
    )
    
    # Load the environment
    env = loader.load(config)
    obs = env.reset()
    return env, obs

def check_requirements():
    """
    Check if the required Android SDK components are available.
    
    Returns:
        dict: Status of each requirement
    """
    requirements = {}
    
    # Check Android SDK root
    sdk_root = os.path.expanduser('~/Android/Sdk')
    requirements['android_sdk_root'] = {
        'path': sdk_root,
        'exists': os.path.exists(sdk_root)
    }
    
    # Check ADB
    adb_path = os.path.expanduser('~/Android/Sdk/platform-tools/adb')
    requirements['adb'] = {
        'path': adb_path,
        'exists': os.path.exists(adb_path)
    }
    
    # Check emulator
    emulator_path = os.path.expanduser('~/Android/Sdk/emulator/emulator')
    requirements['emulator'] = {
        'path': emulator_path,
        'exists': os.path.exists(emulator_path)
    }
    
    # Check AVD home
    avd_home = os.path.expanduser('~/.android/avd')
    requirements['avd_home'] = {
        'path': avd_home,
        'exists': os.path.exists(avd_home)
    }
    
    return requirements

if __name__ == "__main__":
    print("AndroidEnv Wrapper Test")
    print("=" * 50)
    
    # Check requirements first
    print("\nChecking requirements:")
    reqs = check_requirements()
    for name, info in reqs.items():
        status = "✅" if info['exists'] else "❌"
        print(f"  {status} {name}: {info['path']}")
    
    # Try to create environment
    print("\nAttempting to create environment:")
    try:
        env, obs = make_env()
        print("✅ Environment created successfully!")
        print(f"  UI tree length: {len(obs.get('ui_tree', []))}")
        print(f"  Available observation keys: {list(obs.keys())}")
        env.close()
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print("\nTo fix this, you need to:")
        print("1. Install Android Studio")
        print("2. Create an Android Virtual Device (AVD)")
        print("3. Ensure the SDK paths are correct")
        print("4. Set up the environment variables")