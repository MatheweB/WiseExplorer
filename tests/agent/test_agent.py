"""
Tests for wise_explorer.agent.agent

Tests the Agent marker and the State enum.
"""

from wise_explorer.agent.agent import Agent, State


class TestState:
    """State enum tests."""

    def test_all_states_exist(self):
        """All expected states are defined."""
        assert all(hasattr(State, s) for s in ['WIN', 'TIE', 'LOSS', 'NEUTRAL'])

    def test_states_are_distinct(self):
        """Each state has a unique value."""
        values = [State.WIN.value, State.TIE.value, State.LOSS.value, State.NEUTRAL.value]
        assert len(values) == len(set(values))


class TestAgent:
    """Agent is a role-bearing member of the sampling population."""

    def test_default_role(self):
        """An agent defaults to role 0."""
        assert Agent().player_id == 0

    def test_player_id_settable(self):
        """The role can be assigned."""
        agent = Agent()
        agent.player_id = 2
        assert agent.player_id == 2

    def test_agents_are_independent(self):
        """Separate agents don't share their role."""
        a, b = Agent(), Agent()
        a.player_id = 1
        b.player_id = 2
        assert a.player_id != b.player_id
