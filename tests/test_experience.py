import pickle

import pytest

from prlearn import Experience


@pytest.fixture
def experience():
    return Experience()


def test_add_step(experience):
    """Test that add_step correctly adds a single step."""
    experience.add_step(
        observation=1,
        action=2,
        reward=3,
        next_observation=4,
        terminated=True,
        truncated=False,
        info={"key": "value"},
        agent_version=1,
        worker_id=1,
        episode=1,
    )
    assert len(experience) == 1
    assert experience.observations == [1]
    assert experience.actions == [2]
    assert experience.rewards == [3]
    assert experience.next_observations == [4]
    assert experience.terminated == [True]
    assert experience.truncated == [False]
    assert experience.info == [{"key": "value"}]
    assert experience.agent_versions == [1]
    assert experience.worker_ids == [1]
    assert experience.episodes == [1]


def test_clear(experience):
    """Test that clear removes all data from Experience."""
    experience.add_step(1, 2, 3, 4, True, False, {"key": "value"}, 1, 1, 1)
    experience.clear()
    assert len(experience) == 0
    assert experience.observations == []
    assert experience.actions == []
    assert experience.rewards == []
    assert experience.next_observations == []
    assert experience.terminated == []
    assert experience.truncated == []
    assert experience.info == []
    assert experience.agent_versions == []
    assert experience.worker_ids == []
    assert experience.episodes == []


def test_add_experience(experience):
    """Test that add_experience merges another Experience correctly."""
    other_exp = Experience(
        observations=[1],
        actions=[2],
        rewards=[3],
        next_observations=[4],
        terminated=[True],
        truncated=[False],
        info=[{"key": "value"}],
        agent_versions=[1],
        worker_ids=[1],
        episodes=[1],
    )
    experience.add_experience(other_exp)
    assert len(experience) == 1
    assert experience.observations == [1]
    assert experience.actions == [2]
    assert experience.rewards == [3]
    assert experience.next_observations == [4]
    assert experience.terminated == [True]
    assert experience.truncated == [False]
    assert experience.info == [{"key": "value"}]
    assert experience.agent_versions == [1]
    assert experience.worker_ids == [1]
    assert experience.episodes == [1]


def test_copy(experience):
    """Test that copy returns a deep copy of Experience."""
    experience.add_step(1, 2, 3, 4, True, False, {"key": "value"}, 1, 1, 1)
    copied_exp = experience.copy()
    assert copied_exp.observations == [1]
    assert copied_exp.actions == [2]
    assert copied_exp.rewards == [3]
    assert copied_exp.next_observations == [4]
    assert copied_exp.terminated == [True]
    assert copied_exp.truncated == [False]
    assert copied_exp.info == [{"key": "value"}]
    assert copied_exp.agent_versions == [1]
    assert copied_exp.worker_ids == [1]
    assert copied_exp.episodes == [1]
    # Ensure it's a deep copy
    copied_exp.observations[0] = 999
    assert experience.observations[0] == 1


def test_get(experience):
    """Test that get returns correct columns and default columns."""
    experience.add_step(1, 2, 3, 4, True, False, {"key": "value"}, 1, 1, 1)
    data = experience.get(columns=["observations", "actions", "rewards"])
    assert data == ([1], [2], [3])
    default_data = experience.get()
    assert default_data == (
        [1],
        [2],
        [3],
        [4],
        [True],
        [False],
        [{"key": "value"}],
    )


def test_get_experience_batch(experience):
    """Test that get_experience_batch returns the last N steps."""
    for i in range(10):
        experience.add_step(
            i, i + 1, i + 2, i + 3, i % 2 == 0, i % 2 != 0, {"key": i}, i, i, i // 3
        )
    batch = experience.get_experience_batch(5)
    assert batch.observations == [5, 6, 7, 8, 9]
    assert batch.actions == [6, 7, 8, 9, 10]
    assert batch.rewards == [7, 8, 9, 10, 11]
    assert batch.next_observations == [8, 9, 10, 11, 12]
    assert batch.truncated == [True, False, True, False, True]
    assert batch.terminated == [False, True, False, True, False]
    assert batch.info == [{"key": 5}, {"key": 6}, {"key": 7}, {"key": 8}, {"key": 9}]
    assert batch.agent_versions == [5, 6, 7, 8, 9]
    assert batch.worker_ids == [5, 6, 7, 8, 9]
    assert batch.episodes == [1, 2, 2, 2, 3]


def test_empty_experience():
    """Test that an empty Experience behaves as expected."""
    exp = Experience()
    assert len(exp) == 0
    assert exp.get() == ([], [], [], [], [], [], [])
    assert exp.get_experience_batch(5).observations == []


def test_large_experience_stress():
    """Test Experience with a large number of steps (stress test)."""
    exp = Experience()
    n = 10000
    for i in range(n):
        exp.add_step(i, i, i, i, False, False, {}, 0, 0, 0)
    assert len(exp) == n
    batch = exp.get_experience_batch(100)
    assert len(batch.observations) == 100


def test_experience_pickle_serialization():
    """Test that Experience can be pickled and unpickled correctly."""
    exp = Experience()
    exp.add_step(1, 2, 3, 4, True, False, {"key": "value"}, 1, 1, 1)
    data = pickle.dumps(exp)
    loaded = pickle.loads(data)
    assert loaded.observations == [1]
    assert loaded.actions == [2]
    assert loaded.rewards == [3]
    assert loaded.next_observations == [4]
    assert loaded.terminated == [True]
    assert loaded.truncated == [False]
    assert loaded.info == [{"key": "value"}]
    assert loaded.agent_versions == [1]
    assert loaded.worker_ids == [1]
    assert loaded.episodes == [1]


def test_get_with_invalid_columns(experience):
    """Test that get ignores invalid columns and returns only valid ones."""
    experience.add_step(1, 2, 3, 4, True, False, {"key": "value"}, 1, 1, 1)
    data = experience.get(columns=["observations", "not_a_column", "actions"])
    assert data == ([1], [2])


def test_get_experience_batch_zero_size_returns_empty(experience):
    """Requesting a batch of size 0 should return an empty Experience."""
    experience.add_step(1, 2, 3, 4, True, False, {"key": "value"}, 1, 1, 1)
    batch = experience.get_experience_batch(0)
    assert len(batch) == 0
    assert batch.observations == []


def test_get_experience_batch_negative_size_raises(experience):
    """Negative batch sizes are invalid and should raise ValueError."""
    experience.add_step(1, 2, 3, 4, True, False, {"key": "value"}, 1, 1, 1)
    with pytest.raises(ValueError):
        experience.get_experience_batch(-1)


def test_get_experience_batch_non_int_size_raises(experience):
    """Non-integer batch sizes should raise TypeError."""
    experience.add_step(1, 2, 3, 4, True, False, {"key": "value"}, 1, 1, 1)
    with pytest.raises(TypeError):
        experience.get_experience_batch(1.5)


def test_experience_mismatched_lengths_validation():
    """Experience should validate mismatched field lengths on initialization."""
    with pytest.raises(ValueError):
        Experience(observations=[1, 2], actions=[0])
    with pytest.raises(ValueError):
        Experience.from_dict({"observations": [1], "actions": [2, 3]})


def test_experience_json_roundtrip(experience):
    """Experience should serialize to/from JSON without data loss."""
    experience.add_step(1, 2, 3, 4, True, False, {"key": "value"}, 1, 1, 1)
    json_data = experience.to_json()
    loaded = Experience.from_json(json_data)
    assert loaded.observations == experience.observations
    assert loaded.actions == experience.actions
    assert loaded.rewards == experience.rewards
    assert loaded.next_observations == experience.next_observations
    assert loaded.terminated == experience.terminated
    assert loaded.truncated == experience.truncated
    assert loaded.info == experience.info
    assert loaded.agent_versions == experience.agent_versions
    assert loaded.worker_ids == experience.worker_ids
    assert loaded.episodes == experience.episodes


def test_pop_experience_batch_removes_all_when_size_not_provided(experience):
    """pop_experience_batch should return everything and clear the buffer."""
    for i in range(3):
        experience.add_step(i, i + 1, i + 2, i + 3, False, False, {}, i, i, i)
    popped = experience.pop_experience_batch()
    assert len(popped) == 3
    assert len(experience) == 0
    assert popped.rewards == [2, 3, 4]


def test_pop_experience_batch_partial_removal(experience):
    """pop_experience_batch should leave untouched data when size is smaller than buffer."""
    for i in range(5):
        experience.add_step(i, i, i, i, False, False, {}, i, i, i)
    popped = experience.pop_experience_batch(2)
    assert len(popped) == 2
    assert popped.observations == [3, 4]
    assert experience.observations == [0, 1, 2]
    assert len(experience) == 3


if __name__ == "__main__":
    pytest.main()
