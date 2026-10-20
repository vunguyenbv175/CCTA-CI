/**
 * Toolgroup wiring for the multi-STACK viewports.
 *
 * The BASE plan (lib/toolGroupPlan.js) binds Pan (auxiliary), Zoom
 * (secondary) and StackScroll (wheel, CCTA adaptation).  WL_V2 (spec Part
 * 4.2) additionally activates the already-present WindowLevel tool on the
 * PRIMARY drag via WINDOW_LEVEL_DRAG_BINDING: the tool supplies the drag
 * arithmetic ONLY — the client pins each viewport's transfer function to the
 * identity transport ramp (ui/app.js) so the served PNG bytes are never
 * locally re-windowed, and the viewport's window values are submitted as
 * SET_WL_CUSTOM for server rendering.
 *
 * Tool classes are registered ONCE with the tools library (addTool), then the
 * declarative plan (lib/toolGroupPlan.js) is applied to one toolgroup.
 */
import {
  Enums as ToolsEnums,
  ToolGroupManager,
  addTool,
  PanTool,
  StackScrollTool,
  WindowLevelTool,
  ZoomTool,
} from "@cornerstonejs/tools";
import { Settings } from "@cornerstonejs/core";
import {
  MOUSE_BUTTONS,
  WINDOW_LEVEL_DRAG_BINDING,
  applyToolGroupPlan,
  buildToolGroupPlan,
} from "../lib/toolGroupPlan.js";

const MOUSE_BINDINGS = {
  [MOUSE_BUTTONS.PRIMARY]: ToolsEnums.MouseBindings.Primary,
  [MOUSE_BUTTONS.SECONDARY]: ToolsEnums.MouseBindings.Secondary,
  [MOUSE_BUTTONS.AUXILIARY]: ToolsEnums.MouseBindings.Auxiliary,
  [MOUSE_BUTTONS.WHEEL]: ToolsEnums.MouseBindings.Wheel,
};

export const TOOL_CLASSES = {
  WindowLevel: WindowLevelTool,
  Pan: PanTool,
  Zoom: ZoomTool,
  StackScroll: StackScrollTool,
};

let toolsRegistered = false;
const preparedToolGroups = new WeakSet();

/** Register the four CCTA tool classes with the tools library (idempotent). */
export function registerCctaToolClasses() {
  if (toolsRegistered) {
    return;
  }
  for (const ToolClass of Object.values(TOOL_CLASSES)) {
    addTool(ToolClass);
  }
  toolsRegistered = true;
}

/**
 * Ensure the CCTA toolgroup exists, carries the CCTA plan, and is bound to
 * the given viewport (idempotent per toolgroup).
 *
 * @param {{viewportId: string, renderingEngineId: string}} input
 */
export function ensureCctaToolGroup(input) {
  registerCctaToolClasses();
  const plan = buildToolGroupPlan();
  const toolGroup = ToolGroupManager.getToolGroup(plan.groupId) ||
    ToolGroupManager.createToolGroup(plan.groupId);
  if (!toolGroup) {
    throw new Error("CCTA toolgroup could not be created");
  }
  if (!preparedToolGroups.has(toolGroup)) {
    // The admitted WindowLevel tool's SVG cursor is a blob: URL, which the
    // frozen `default-src 'self'` CSP refuses (a security violation on every
    // viewport add).  This client has never used tool cursors; disable them
    // so the v2 activation stays inside the unchanged CSP.
    Settings.getRuntimeSettings().set("useCursors", false);
    applyToolGroupPlan(toolGroup, plan, { mouseBindings: MOUSE_BINDINGS });
    // WL_V2 (spec Part 4.2): activate the primary-drag WindowLevel binding on
    // top of the closed base plan.  The tool's viewport VOI is neutralized to
    // the identity transport mapping by ui/app.js; only its drag arithmetic
    // reaches the server (SET_WL_CUSTOM posts).
    toolGroup.setToolActive(WINDOW_LEVEL_DRAG_BINDING.tool, {
      bindings: [
        { mouseButton: MOUSE_BINDINGS[WINDOW_LEVEL_DRAG_BINDING.mouseButton] },
      ],
    });
    preparedToolGroups.add(toolGroup);
  }
  toolGroup.addViewport(input.viewportId, input.renderingEngineId);
  return toolGroup;
}

export default { ensureCctaToolGroup, registerCctaToolClasses };
