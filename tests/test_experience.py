import pytest

from prlearn import Experience


@pytest.fixture
def experience():
    return Experience()


def test_add_step(experience):
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


def test_get(experience):
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


if __name__ == "__main__":
    pytest.main()
