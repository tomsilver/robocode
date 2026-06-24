"""Tests for robocode.prompts composition helpers and blackbox deltas."""

from robocode import prompts
from robocode.mcp import (
    MCP_TOOLS_SYSTEM_PROMPT_SUFFIX,
    MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX,
)
from robocode.utils.backends import CLAUDE_PROMPT_SUFFIX, OPENCODE_PROMPT_SUFFIX

# ---------------------------------------------------------------------------
# Composed intros: the shared blackbox-discovery clause is one fragment, not a
# per-approach copy.
# ---------------------------------------------------------------------------


def test_blackbox_discovery_clause_shared_between_intros():
    """The blackbox-discovery clause is shared; identities genuinely differ."""
    assert "discover the dynamics" in prompts.AGENTIC_INTRO_BLACKBOX
    assert "discover the dynamics" in prompts.CDL_INTRO_BLACKBOX
    assert "black box" not in prompts.AGENTIC_INTRO
    assert "black box" not in prompts.CDL_INTRO
    assert "purely imperative, feedforward" in prompts.CDL_INTRO
    assert "purely imperative, feedforward" not in prompts.AGENTIC_INTRO


def test_learning_clause_shared_across_approaches():
    """Both approaches get the same learning clause; only blackbox wording differs."""
    # The non-blackbox learning clause is identical for agentic and CDL.
    assert "read environment source code" in prompts.AGENTIC_INTRO
    assert "read environment source code" in prompts.CDL_INTRO
    # All four intros state the agent writes an approach class.
    for intro in (
        prompts.AGENTIC_INTRO,
        prompts.AGENTIC_INTRO_BLACKBOX,
        prompts.CDL_INTRO,
        prompts.CDL_INTRO_BLACKBOX,
    ):
        assert "write an optimal approach class" in intro


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


def test_system_prompt_agentic_non_blackbox():
    """Non-blackbox agentic system prompt reads source and ends with the suffix."""
    sp = prompts.build_system_prompt(
        intro=prompts.AGENTIC_INTRO, blackbox=False, backend_name="claude"
    )
    assert "expert at writing policies" in sp
    assert "VERSION CONTROL" in sp
    assert "explore source code in parallel" in sp
    assert "black box" not in sp
    assert sp.endswith(CLAUDE_PROMPT_SUFFIX)
    assert MCP_TOOLS_SYSTEM_PROMPT_SUFFIX not in sp


def test_system_prompt_blackbox_swaps_subagents():
    """Blackbox selects the empirical-exploration subagent guidance."""
    sp = prompts.build_system_prompt(
        intro=prompts.AGENTIC_INTRO_BLACKBOX, blackbox=True, backend_name="claude"
    )
    assert "black box" in sp
    assert "run exploration experiments" in sp
    assert "explore source code in parallel" not in sp


def test_system_prompt_opencode_backend_suffix():
    """The opencode backend swaps in its own prompt suffix."""
    sp = prompts.build_system_prompt(
        intro=prompts.AGENTIC_INTRO, blackbox=False, backend_name="opencode"
    )
    assert sp.endswith(OPENCODE_PROMPT_SUFFIX)
    assert CLAUDE_PROMPT_SUFFIX not in sp


def test_system_prompt_mcp_suffix_blackbox_variant():
    """Blackbox + mcp tools appends the blackbox MCP suffix, not the normal one."""
    sp = prompts.build_system_prompt(
        intro=prompts.AGENTIC_INTRO_BLACKBOX,
        blackbox=True,
        backend_name="claude",
        mcp_tools=("render_state",),
    )
    assert MCP_TOOLS_SYSTEM_PROMPT_SUFFIX_BLACKBOX in sp
    assert MCP_TOOLS_SYSTEM_PROMPT_SUFFIX not in sp


def test_system_prompt_mcp_suffix_non_blackbox_variant():
    """Non-blackbox + mcp tools appends the standard MCP suffix."""
    sp = prompts.build_system_prompt(
        intro=prompts.AGENTIC_INTRO,
        blackbox=False,
        backend_name="claude",
        mcp_tools=("render_state",),
    )
    assert MCP_TOOLS_SYSTEM_PROMPT_SUFFIX in sp


def test_system_prompt_token_budget_on_both_approaches():
    """Token-budget guidance is appended to every system prompt (both approaches)."""
    agentic = prompts.build_system_prompt(
        intro=prompts.AGENTIC_INTRO, blackbox=False, backend_name="claude"
    )
    cdl = prompts.build_system_prompt(
        intro=prompts.CDL_INTRO, blackbox=False, backend_name="claude"
    )
    assert "TOKEN BUDGET" in agentic
    assert "TOKEN BUDGET" in cdl


# ---------------------------------------------------------------------------
# build_interface_spec
# ---------------------------------------------------------------------------


def test_interface_spec_non_blackbox_appends_inspect():
    """Non-blackbox interface spec appends the inspect-source suffix."""
    spec = prompts.build_interface_spec(
        class_interface=prompts.AGENTIC_CLASS_INTERFACE,
        run_commands=prompts.AGENTIC_RUN_COMMANDS,
        python_executable="/usr/bin/python3",
        primitives_description="PRIMS_DESC",
        blackbox=False,
    )
    assert "inspect the source code" in spec
    assert "/usr/bin/python3 test_approach.py" in spec
    assert "PRIMS_DESC" in spec


def test_interface_spec_blackbox_omits_inspect():
    """Blackbox interface spec omits the inspect-source suffix."""
    spec = prompts.build_interface_spec(
        class_interface=prompts.AGENTIC_CLASS_INTERFACE,
        run_commands=prompts.AGENTIC_RUN_COMMANDS,
        python_executable="/usr/bin/python3",
        primitives_description="PRIMS_DESC",
        blackbox=True,
    )
    assert "inspect the source code" not in spec
    assert "PRIMS_DESC" in spec


def test_interface_spec_shared_wrapper_across_approaches():
    """Both approaches share the interface-spec wrapper, including the test note."""
    common = {
        "python_executable": "/py",
        "primitives_description": "PRIMS",
        "blackbox": False,
    }
    agentic = prompts.build_interface_spec(
        class_interface=prompts.AGENTIC_CLASS_INTERFACE,
        run_commands=prompts.AGENTIC_RUN_COMMANDS,
        **common,
    )
    cdl = prompts.build_interface_spec(
        class_interface=prompts.CDL_CLASS_INTERFACE,
        run_commands=prompts.CDL_RUN_COMMANDS,
        **common,
    )
    for spec in (agentic, cdl):
        assert "Write the best approach you can" in spec
        assert "Write test scripts that use the real environment" in spec
    # Structural slots still differ.
    assert "test_behavior_[behavior_name].py" in cdl
    assert "test_behavior_[behavior_name].py" not in agentic


# ---------------------------------------------------------------------------
# build_agentic_prompt
# ---------------------------------------------------------------------------


def test_agentic_prompt_blackbox():
    """Blackbox agentic prompt warns about env_client and includes geometry."""
    p = prompts.build_agentic_prompt(
        blackbox=True,
        interface_spec="IFACE",
        geometry=True,
        modular_code=False,
        env_description=None,
    )
    assert "BLACK BOX" in p
    assert "must NOT import `env_client`" in p
    assert "Read the environment source files" not in p
    assert "5-10 bullet points" in p
    assert "IFACE" in p


def test_agentic_prompt_blackbox_with_description_and_no_geometry():
    """Blackbox prompt wraps the description and omits geometry when disabled."""
    p = prompts.build_agentic_prompt(
        blackbox=True,
        interface_spec="IFACE",
        geometry=False,
        modular_code=False,
        env_description="ENVDESC",
    )
    assert "The environment is described below." in p
    assert "ENVDESC" in p
    assert "5-10 bullet points" not in p


def test_agentic_prompt_with_description_non_blackbox():
    """Non-blackbox prompt with a description never asks to read source."""
    p = prompts.build_agentic_prompt(
        blackbox=False,
        interface_spec="IFACE",
        geometry=True,
        modular_code=True,
        env_description="ENVDESC",
    )
    assert "ENVDESC" in p
    assert "BLACK BOX" not in p
    assert "Read the environment source files" not in p
    assert "Write MODULAR code" in p


def test_agentic_prompt_source_non_blackbox():
    """Non-blackbox prompt without a description asks to read source, with geometry."""
    p = prompts.build_agentic_prompt(
        blackbox=False,
        interface_spec="IFACE",
        geometry=True,
        modular_code=False,
        env_description=None,
    )
    assert "Read the environment source files" in p
    assert "5-10 bullet points" in p  # geometry applies in every branch
    assert "Write MODULAR code" not in p


# ---------------------------------------------------------------------------
# build_cdl_prompt
# ---------------------------------------------------------------------------


def test_cdl_prompt_blackbox():
    """Blackbox CDL prompt uses the empirical obs note and the canonical D1 spec."""
    p = prompts.build_cdl_prompt(
        blackbox=True,
        interface_spec="IFACE",
        geometry=True,
        env_description=None,
        has_initial_helpers=False,
    )
    assert "BLACK BOX" in p
    assert "must NOT import `env_client`" in p
    assert "map the observation layout empirically" in p
    assert "devectorize" in p
    assert "Read the environment source files" not in p
    # D1: the behavior-precondition set_state note and the code-block
    # make_primitives form (canonical, shared with the monolithic approach).
    assert "precondition requires when testing it in" in p
    assert "primitives = env.make_primitives()" in p
    assert "ALREADY PROVIDED" not in p


def test_geometry_prompt_unified_across_approaches():
    """Both approaches use the same (terse) geometry-reasoning prompt."""
    agentic = prompts.build_agentic_prompt(
        blackbox=False,
        interface_spec="IFACE",
        geometry=True,
        modular_code=False,
        env_description="E",
    )
    cdl = prompts.build_cdl_prompt(
        blackbox=False,
        interface_spec="IFACE",
        geometry=True,
        env_description="E",
        has_initial_helpers=False,
    )
    assert prompts.GEOMETRY_PROMPT in agentic
    assert prompts.GEOMETRY_PROMPT in cdl
    assert "5-10 bullet points" in agentic
    assert "5-10 bullet points" in cdl


def test_cdl_prompt_initial_helpers_non_blackbox():
    """Non-blackbox CDL prompt with helpers uses the devectorize obs note."""
    p = prompts.build_cdl_prompt(
        blackbox=False,
        interface_spec="IFACE",
        geometry=True,
        env_description="ENVDESC",
        has_initial_helpers=True,
    )
    assert "ENVDESC" in p
    assert "ALREADY PROVIDED" in p
    assert "devectorize" in p
    assert "map the observation layout empirically" not in p
    assert "BLACK BOX" not in p


# ---------------------------------------------------------------------------
# build_mcp_tool_lines
# ---------------------------------------------------------------------------


def test_mcp_tool_lines_empty():
    """No MCP tools yields an empty string."""
    assert (
        prompts.build_mcp_tool_lines(
            mcp_tools=(), backend_name="claude", blackbox=False
        )
        == ""
    )


def test_mcp_tool_lines_claude_naming():
    """Claude backend uses the mcp__server__tool naming."""
    lines = prompts.build_mcp_tool_lines(
        mcp_tools=("render_state",), backend_name="claude", blackbox=False
    )
    assert "MCP tools for visual debugging" in lines
    assert "mcp__robocode-tools__render_state" in lines


def test_mcp_tool_lines_opencode_naming():
    """OpenCode backend uses the server_tool naming."""
    lines = prompts.build_mcp_tool_lines(
        mcp_tools=("render_state",), backend_name="opencode", blackbox=False
    )
    assert "robocode-tools_render_state" in lines


# ---------------------------------------------------------------------------
# genplan constants (kept faithful to upstream llm-genplan)
# ---------------------------------------------------------------------------


def test_genplan_constants():
    """Genplan prompt constants carry their distinctive single-code-block text."""
    assert "GeneratedApproach" in prompts.GENPLAN_INTERFACE_SPEC
    assert "Return ONLY" in prompts.GENPLAN_INTERFACE_SPEC
    assert prompts.GENPLAN_SUMMARY_PROMPT
    assert "simple strategy" in prompts.GENPLAN_STRATEGY_PROMPT
