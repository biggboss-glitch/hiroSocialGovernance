"""
Pre-submission validation script for Hiro Social Governance Environment.

Run this before submitting to ensure all requirements are met.
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding to handle any Unicode output
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"  [PASS] {description}: {filepath}")
        return True
    else:
        print(f"  [FAIL] {description} MISSING: {filepath}")
        return False


def check_required_files() -> bool:
    """Check all required files exist."""
    print("\n" + "=" * 60)
    print("CHECKING REQUIRED FILES")
    print("=" * 60)

    required = [
        ("openenv.yaml", "OpenEnv spec file"),
        ("Dockerfile", "Docker configuration"),
        ("requirements.txt", "Python dependencies"),
        ("README.md", "Documentation"),
        ("inference.py", "Baseline inference script"),
        ("src/__init__.py", "Source package"),
        ("src/environment.py", "Core environment"),
        ("src/models.py", "Pydantic models"),
        ("src/agents.py", "Agent system"),
        ("src/dynamics.py", "Social dynamics"),
        ("src/reward.py", "Reward calculation"),
        ("src/tasks/__init__.py", "Tasks package"),
        ("api/server.py", "API server"),
        ("LICENSE", "License file"),
    ]

    all_exist = True
    for filepath, description in required:
        if not check_file_exists(filepath, description):
            all_exist = False

    return all_exist


def check_openenv_yaml() -> bool:
    """Validate openenv.yaml structure."""
    print("\n" + "=" * 60)
    print("CHECKING openenv.yaml")
    print("=" * 60)

    try:
        import yaml

        with open("openenv.yaml", "r") as f:
            spec = yaml.safe_load(f)

        required_keys = ["name", "version", "description", "environment", "models", "tasks"]
        for key in required_keys:
            if key not in spec:
                print(f"  [FAIL] Missing key: {key}")
                return False
            print(f"  [PASS] Has key: {key}")

        # Check tasks
        tasks = spec.get("tasks", [])
        if len(tasks) < 3:
            print(f"  [FAIL] Need at least 3 tasks, found {len(tasks)}")
            return False
        print(f"  [PASS] Has {len(tasks)} tasks")

        # Check reward formula
        if "reward_formula" not in spec:
            print("  [FAIL] Missing reward_formula")
            return False
        print("  [PASS] Has reward_formula")

        return True

    except ImportError:
        print("  [WARN] PyYAML not installed, skipping YAML validation")
        return True
    except Exception as e:
        print(f"  [FAIL] Error validating openenv.yaml: {e}")
        return False


def check_models() -> bool:
    """Check Pydantic models can be imported."""
    print("\n" + "=" * 60)
    print("CHECKING MODELS")
    print("=" * 60)

    try:
        from src.models import Observation, Action, Reward, ActionType, AgentType
        print("  [PASS] Models import successfully")

        # Test observation creation
        obs = Observation(
            step=1,
            max_steps=100,
            num_agents=10,
            active_agents=10
        )
        print("  [PASS] Observation can be created")

        # Test action creation
        action = Action(action_type=ActionType.NO_ACTION)
        print("  [PASS] Action can be created")

        # Test reward creation
        reward = Reward(total=0.5)
        print("  [PASS] Reward can be created")

        return True

    except Exception as e:
        print(f"  [FAIL] Error with models: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_environment() -> bool:
    """Check environment can be instantiated and used."""
    print("\n" + "=" * 60)
    print("CHECKING ENVIRONMENT")
    print("=" * 60)

    try:
        from src.environment import HiroSocialEnv
        from src.models import Action, ActionType

        # Create environment
        env = HiroSocialEnv()
        print("  [PASS] Environment can be created")

        # Test each task
        tasks = ["easy", "medium", "hard"]

        for task_id in tasks:
            print(f"\n  Testing {task_id}...")

            # Reset
            obs = env.reset(task=task_id, seed=42)
            print(f"    [PASS] reset() works")

            # Take one step
            action = Action(action_type=ActionType.NO_ACTION)
            obs, reward, done, info = env.step(action)
            print(f"    [PASS] step() works")

            # Check state
            state = env.state()
            print(f"    [PASS] state() works")

            # Check reward range
            if not (-1.0 <= reward.total <= 1.0):
                print(f"    [FAIL] Reward out of range: {reward.total}")
                return False

            print(f"    [PASS] Reward in range: {reward.total:.3f}")

        return True

    except Exception as e:
        print(f"  [FAIL] Error with environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_graders() -> bool:
    """Check graders produce valid scores."""
    print("\n" + "=" * 60)
    print("CHECKING GRADERS")
    print("=" * 60)

    try:
        from src.environment import HiroSocialEnv
        from src.models import Action, ActionType

        tasks = ["easy", "medium", "hard"]

        for task_id in tasks:
            print(f"\n  Testing {task_id} grader...")

            env = HiroSocialEnv()
            env.reset(task=task_id, seed=42)

            # Run full episode
            done = False
            steps = 0
            while not done:
                action = Action(action_type=ActionType.NO_ACTION)
                _, _, done, _ = env.step(action)
                steps += 1

            # Get grade
            grade = env.get_task_grade()

            if not (0.0 <= grade <= 1.0):
                print(f"    [FAIL] Grade out of range: {grade}")
                return False

            print(f"    [PASS] Grade in range: {grade:.3f} ({steps} steps)")

        return True

    except Exception as e:
        print(f"  [FAIL] Error with graders: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_inference_script() -> bool:
    """Check inference script exists and is valid Python."""
    print("\n" + "=" * 60)
    print("CHECKING INFERENCE SCRIPT")
    print("=" * 60)

    try:
        # Check syntax
        with open("inference.py", "r") as f:
            code = f.read()

        compile(code, "inference.py", "exec")
        print("  [PASS] inference.py is valid Python")

        # Check imports
        from inference import GovernanceAgent, run_task
        print("  [PASS] inference.py imports successfully")

        return True

    except SyntaxError as e:
        print(f"  [FAIL] Syntax error in inference.py: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error with inference.py: {e}")
        return False


def check_dockerfile() -> bool:
    """Check Dockerfile is valid."""
    print("\n" + "=" * 60)
    print("CHECKING DOCKERFILE")
    print("=" * 60)

    try:
        with open("Dockerfile", "r") as f:
            content = f.read()

        # Check for required elements
        required = ["FROM", "WORKDIR", "COPY", "RUN", "EXPOSE", "CMD"]
        for elem in required:
            if elem not in content:
                print(f"  [FAIL] Missing {elem} in Dockerfile")
                return False
            print(f"  [PASS] Has {elem}")

        # Check port
        if "7860" not in content:
            print("  [FAIL] Port 7860 not exposed")
            return False
        print("  [PASS] Exposes port 7860")

        # Check health check
        if "HEALTHCHECK" not in content:
            print("  [WARN] No HEALTHCHECK (recommended but not required)")
        else:
            print("  [PASS] Has HEALTHCHECK")

        return True

    except Exception as e:
        print(f"  [FAIL] Error checking Dockerfile: {e}")
        return False


def check_api_server() -> bool:
    """Check API server can be imported and endpoints exist."""
    print("\n" + "=" * 60)
    print("CHECKING API SERVER")
    print("=" * 60)

    try:
        from api.server import app
        print("  [PASS] API server imports successfully")

        # Check routes exist
        routes = [route.path for route in app.routes]
        required_routes = ["/health", "/reset", "/step", "/state", "/tasks", "/grade"]
        for route in required_routes:
            if route in routes:
                print(f"  [PASS] Has endpoint: {route}")
            else:
                print(f"  [FAIL] Missing endpoint: {route}")
                return False

        return True

    except Exception as e:
        print(f"  [FAIL] Error with API server: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("HIRO SOCIAL GOVERNANCE - PRE-SUBMISSION VALIDATION")
    print("=" * 60)

    checks = [
        ("Required Files", check_required_files),
        ("openenv.yaml", check_openenv_yaml),
        ("Models", check_models),
        ("Environment", check_environment),
        ("Graders", check_graders),
        ("Inference Script", check_inference_script),
        ("Dockerfile", check_dockerfile),
        ("API Server", check_api_server),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  [FAIL] {name} check failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(results)} checks")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n" + "=" * 60)
        print("  ALL CHECKS PASSED - READY FOR SUBMISSION")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("  SOME CHECKS FAILED - FIX BEFORE SUBMITTING")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
