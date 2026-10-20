/**
 * Declarative Cornerstone toolgroup plan (ADAPT of the CMR tools.ts bindings).
 *
 * CCTA adaptations, deliberately different from the CMR source:
 *   - Wheel -> StackScroll: the CCTA client binds the wheel to stack scroll
 *     because the frozen per-plane frame endpoints serve scrollable stacks.
 *     This is a CCTA adaptation, NOT an observed CMR binding (prior art E3).
 *   - WindowLevel is part of the base add-set but the BASE plan carries NO
 *     active WL binding.  The base plan is the frozen v1 CS-001 surface
 *     (unit-pinned: passive == [WindowLevel]); the v2 custom-WL successor
 *     (spec Part 4.2) activates the drag binding EXPLICITLY in tools.js via
 *     WINDOW_LEVEL_DRAG_BINDING after the base plan is applied.  The drag
 *     path only supplies interaction arithmetic: the client pins the
 *     viewport's transfer function to the identity transport ramp (ui/app.js)
 *     so the served PNG bytes are never locally re-windowed, and every WL
 *     value is submitted as SET_WL_CUSTOM for server rendering.
 *
 * The plan is data (testable without the tools library); applyToolGroupPlan
 * is the thin adapter over ToolGroupManager.
 */
export const TOOL_GROUP_ID = "ccta-viewer-toolgroup";

export const TOOL_NAMES = Object.freeze({
  WINDOW_LEVEL: "WindowLevel",
  PAN: "Pan",
  ZOOM: "Zoom",
  STACK_SCROLL: "StackScroll",
});

export const MOUSE_BUTTONS = Object.freeze({
  PRIMARY: "Primary",
  SECONDARY: "Secondary",
  AUXILIARY: "Auxiliary",
  WHEEL: "Wheel",
});

/**
 * The v2 custom-WL drag binding (primary button), applied by tools.js on top
 * of the base plan.  Kept as data here so the binding stays testable without
 * the tools library.
 */
export const WINDOW_LEVEL_DRAG_BINDING = Object.freeze({
  tool: TOOL_NAMES.WINDOW_LEVEL,
  mouseButton: MOUSE_BUTTONS.PRIMARY,
});

/**
 * @returns {{
 *   groupId: string,
 *   add: string[],
 *   passive: string[],
 *   active: Array<{tool: string, mouseButton: string}>,
 * }}
 */
export function buildToolGroupPlan() {
  return {
    groupId: TOOL_GROUP_ID,
    add: [
      TOOL_NAMES.WINDOW_LEVEL,
      TOOL_NAMES.PAN,
      TOOL_NAMES.ZOOM,
      TOOL_NAMES.STACK_SCROLL,
    ],
    // Present in the base add-set; the BASE plan activates no WL binding.
    // tools.js applies the v2 WINDOW_LEVEL_DRAG_BINDING on top (spec 4.2).
    passive: [TOOL_NAMES.WINDOW_LEVEL],
    active: [
      { tool: TOOL_NAMES.PAN, mouseButton: MOUSE_BUTTONS.AUXILIARY },
      { tool: TOOL_NAMES.ZOOM, mouseButton: MOUSE_BUTTONS.SECONDARY },
      { tool: TOOL_NAMES.STACK_SCROLL, mouseButton: MOUSE_BUTTONS.WHEEL },
    ],
  };
}

/**
 * Apply a toolgroup plan through the real ToolGroupManager surface.
 *
 * @param {object} toolGroup a Cornerstone ToolGroup
 * @param {ReturnType<typeof buildToolGroupPlan>} plan
 * @param {{mouseBindings: Record<string, number>, viewportId?: string, renderingEngineId?: string}} options
 */
export function applyToolGroupPlan(toolGroup, plan, options) {
  const { mouseBindings } = options;
  for (const toolName of plan.add) {
    toolGroup.addTool(toolName);
  }
  for (const toolName of plan.passive) {
    toolGroup.setToolPassive(toolName);
  }
  for (const binding of plan.active) {
    const mouseButton = mouseBindings[binding.mouseButton];
    if (mouseButton === undefined) {
      throw new Error(`toolgroup plan references unknown binding ${binding.mouseButton}`);
    }
    toolGroup.setToolActive(binding.tool, { bindings: [{ mouseButton }] });
  }
  if (options.viewportId && options.renderingEngineId) {
    toolGroup.addViewport(options.viewportId, options.renderingEngineId);
  }
  return toolGroup;
}

export default { TOOL_GROUP_ID, TOOL_NAMES, WINDOW_LEVEL_DRAG_BINDING, buildToolGroupPlan, applyToolGroupPlan };
