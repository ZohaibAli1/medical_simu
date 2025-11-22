from enum import Enum

class ProcedureAction(Enum):
    INCISION = "incision"
    DISSECTION = "dissection"
    CAUTERIZE = "cauterize"
    SUTURE = "suture"
    EXTRACT = "extract"
    SELECT_TOOL = "select_tool"
    IDENTIFY_ORGAN = "identify_organ"
    NONE = "none"  # Fixed: Added NONE

class ProcedureStep:
    def __init__(self, step_id, description, required_action, target_organ=None, required_tool=None, success_criteria=None):
        self.step_id = step_id
        self.description = description
        self.required_action = required_action
        self.target_organ = target_organ
        self.required_tool = required_tool
        self.success_criteria = success_criteria if success_criteria is not None else {}
        self.is_completed = False

    def check_completion(self, action_data):
        # Simple check: action type must match
        if action_data['action'] != self.required_action:
            return False

        # More complex checks based on success criteria
        if self.target_organ and action_data.get('target') != self.target_organ:
            return False

        self.is_completed = True
        return True

class ProcedureScript:
    def __init__(self, name, steps):
        self.name = name
        self.steps = steps
        self.current_step_index = 0

    def get_current_step(self):
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance_step(self):
        if self.current_step_index < len(self.steps):
            self.current_step_index += 1
            return self.get_current_step()
        return None

    def reset(self):
        self.current_step_index = 0
        for step in self.steps:
            step.is_completed = False

# --- Corrected Function Definition ---
def create_appendectomy_script():
    steps = [
        ProcedureStep(
            step_id=1,
            description="Identify the appendix and surrounding structures.",
            required_action=ProcedureAction.IDENTIFY_ORGAN,
            target_organ="appendix",
            required_tool="none"
        ),
        ProcedureStep(
            step_id=2,
            description="Make a small incision over the appendix area.",
            required_action=ProcedureAction.INCISION,
            required_tool="scalpel",
            success_criteria={'length_range': (50, 80), 'depth': 1}
        ),
        ProcedureStep(
            step_id=3,
            description="Carefully dissect the appendix from the surrounding tissue.",
            required_action=ProcedureAction.DISSECTION,
            target_organ="appendix",
            required_tool="forceps"
        ),
        ProcedureStep(
            step_id=4,
            description="Cauterize the base of the appendix.",
            required_action=ProcedureAction.CAUTERIZE,
            target_organ="appendix_base",
            required_tool="cauterizer"
        ),
        ProcedureStep(
            step_id=5,
            description="Extract the appendix.",
            required_action=ProcedureAction.EXTRACT,
            target_organ="appendix",
            required_tool="forceps"
        ),
        ProcedureStep(
            step_id=6,
            description="Suture the incision site.",
            required_action=ProcedureAction.SUTURE,
            required_tool="needle_driver"
        )
    ]
    return ProcedureScript("Appendectomy", steps)

def create_gallbladder_script():
    steps = [
        ProcedureStep(
            step_id=1,
            description="Identify the gallbladder.",
            required_action=ProcedureAction.IDENTIFY_ORGAN,
            target_organ="gallbladder",
            required_tool="none"
        ),
        ProcedureStep(
            step_id=2,
            description="Dissect the cystic duct.",
            required_action=ProcedureAction.DISSECTION,
            target_organ="cystic_duct",
            required_tool="forceps"
        ),
        ProcedureStep(
            step_id=3,
            description="Clip and cut the cystic duct.",
            required_action=ProcedureAction.INCISION,
            target_organ="cystic_duct",
            required_tool="scalpel"
        ),
        ProcedureStep(
            step_id=4,
            description="Extract the gallbladder.",
            required_action=ProcedureAction.EXTRACT,
            target_organ="gallbladder",
            required_tool="forceps"
        )
    ]
    return ProcedureScript("Gallbladder Removal", steps)

if __name__ == '__main__':
    script = create_appendectomy_script()
    print(f"Procedure: {script.name}")
    script2 = create_gallbladder_script()
    print(f"Procedure: {script2.name}")