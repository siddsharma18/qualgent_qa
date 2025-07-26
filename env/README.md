# Android Environment Wrapper

This directory contains a wrapper for the AndroidEnv environment for Android automation tasks.

## Files

- `android_env_wrapper.py`: Main wrapper module for creating AndroidEnv instances
- `simple_task.textproto`: A simple task configuration file for testing
- `README.md`: This documentation file

## Setup Requirements

To use the AndroidEnv wrapper, you need:

1. **Android Studio** installed
2. **Android SDK** configured
3. **Android Virtual Device (AVD)** created
4. **ADB (Android Debug Bridge)** available
5. **Emulator** available

### Installation Steps

1. Install Android Studio from [https://developer.android.com/studio](https://developer.android.com/studio)
2. Open Android Studio and go to "AVD Manager"
3. Create a new Virtual Device (e.g., Pixel 2 API 30)
4. Ensure the SDK paths are correctly set:
   - Android SDK Root: `~/Android/Sdk`
   - AVD Home: `~/.android/avd`
   - Platform Tools: `~/Android/Sdk/platform-tools`
   - Emulator: `~/Android/Sdk/emulator`

### Environment Variables

Set these environment variables (optional, the wrapper uses default paths):

```bash
export ANDROID_SDK_ROOT=~/Android/Sdk
export ANDROID_AVD_HOME=~/.android/avd
```

## Usage

```python
from env.android_env_wrapper import make_env

# Create environment with default task
env, obs = make_env()

# Create environment with custom task file
env, obs = make_env(task_path="path/to/your/task.textproto")

# Check requirements
from env.android_env_wrapper import check_requirements
reqs = check_requirements()
```

## Testing

Run the wrapper test:

```bash
python env/android_env_wrapper.py
```

This will check all requirements and attempt to create an environment.

## Task Configuration

The `simple_task.textproto` file contains a basic task configuration that:
- Launches the Android Settings app
- Sets up the expected app screen
- Provides reset steps

You can create custom task files following the AndroidEnv proto format.

## Troubleshooting

1. **"No such file or directory: adb"**: Install Android SDK and ensure ADB is in the correct path
2. **"AVD not found"**: Create an Android Virtual Device in Android Studio
3. **"Emulator not found"**: Ensure the emulator is installed with the SDK
4. **Task configuration errors**: Check the proto format in your task file 