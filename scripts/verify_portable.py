"""Verify core imports and deterministic CPU episodes without model downloads."""
from mindscape.environments.multiplication.environment import MultiplicationEnvironment
from mindscape.verification.verifier import verify

def main():
    for left, right in [(19, 3), (-123, 45), (0, 7), (999, 99)]:
        env = MultiplicationEnvironment()
        env.reset((left, right))
        trajectory = env.run()
        answer = env.get_state().answer
        assert answer == left * right
        assert env.is_goal_reached()
        assert verify(trajectory, answer).verified
    print('PASS: four CPU episodes and independent trajectory verification')

if __name__ == '__main__':
    main()
