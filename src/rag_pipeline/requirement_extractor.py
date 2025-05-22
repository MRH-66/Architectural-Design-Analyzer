# ArchitecturalRAGSystem/src/rag_pipeline/requirement_extractor.py
import google.generativeai as genai
import json
from typing import Dict, Any, Optional, List, Union
import os


# It's good practice to import your config to get model names and API key
# Ensure your config.py and .env are set up
# from src.config import Config

class RequirementExtractor:
    """
    Extracts structured requirements from a user conversation JSON using Gemini.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initializes the RequirementExtractor.

        Args:
            model_name (str): The name of the Gemini model to use for extraction
                              (e.g., "models/gemini-1.5-flash-latest").
            api_key (Optional[str]): The Google API Key. If None, assumes
                                     genai.configure() has been called.
        """
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)
        # It's assumed genai.configure() has been called if api_key is None.

        # Initialize the generative model
        try:
            self.model = genai.GenerativeModel(self.model_name)
            print(
                f"RequirementExtractor: Initialized Gemini model '{self.model_name}'.")
        except Exception as e:
            print(f"Error initializing Gemini model '{self.model_name}': {e}")
            self.model = None  # Set to None if initialization fails

    def _create_extraction_prompt(self, conversation_json_string: str) -> str:
        """
        Creates a detailed prompt for Gemini to extract requirements.
        """
        # This prompt is crucial and will need iteration and refinement.
        prompt = f"""
        Analyze the following conversation JSON, which represents an interview between a user and a floor plan designer.
        The goal is to extract key requirements and preferences for designing a house floor plan and preparing initial BOQs.

        Conversation JSON:
        ```json
        {conversation_json_string}
        ```

        Extract the following information and structure it as a JSON object:
        1.  "project_summary":
            *   "building_type": (e.g., "House", "Apartment")
            *   "total_footprint_sqft": (Approximate square footage if mentioned, can be integer or null)
            *   "num_floors": (Number of floors above ground, can be integer or null)
            *   "num_basements": (Number of basements, can be integer or null)
            *   "user_style_preference": (A short string describing style, e.g., "Modern, Minimalist, Aesthetic, Rich", "Traditional")
            *   "budget_level": (e.g., "Moderate", "High-end", "Budget-conscious", "Not specified")
            *   "key_constraints_or_desires": (A list of strings for major non-room specific desires like "Minimize direct sunlight but maximize natural light", "Private backyard")
        2.  "room_specifications": (A list of dictionaries, one for each distinct room or area requested)
            *   "room_name": (e.g., "Master Bedroom", "Kitchen", "Living Area", "Gym", "Garage Bathroom", "Backyard Bathroom")
            *   "quantity": (Default 1, or as specified if multiple similar rooms are requested, e.g. 4 for "4 bedrooms")
            *   "attributes": (A list of strings describing specific features or requirements for that room, e.g., "attached bath", "hidden door to basement", "not open to living room", "south-facing with shading")
            *   "connectivity_notes": (A list of strings describing how this room connects to others, e.g., "connects to living area", "access from master bedroom only")
        3.  "special_features": (A list of dictionaries for unique or non-standard features)
            *   "feature_name": (e.g., "Hidden Basement Access", "Baithak / Transition Sitting Area", "Outdoor Pool & Bonfire Area")
            *   "description": (Brief description or user's verbatim request for it)
            *   "related_rooms": (List of rooms this feature is associated with, if any)
        4.  "site_and_orientation":
            *   "lot_shape": (e.g., "Rectangular", "Square", "Irregular", "Not specified")
            *   "lot_orientation_street_facing": (e.g., "North", "South", "East", "West", "Not specified")
            *   "lot_width_vs_depth": (e.g., "Wider than deep (shorter side faces street)", "Deeper than wide (longer side faces street)", "Not specified")

        If information for a field is not present in the conversation, use `null` for single values or an empty list `[]` for lists.
        Be precise and extract information as directly as possible from the conversation.
        If the conversation JSON contains an embedded JSON (e.g., from an image analysis), incorporate that information as if it were part of the textual conversation.
        Focus on user statements about what they *want* or *need*.

        Output ONLY the JSON object. Do not include any other text before or after the JSON.
        """
        return prompt

    def extract_requirements(self, conversation_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """
        Extracts requirements from the provided conversation data.

        Args:
            conversation_data (Union[Dict, List]): The conversation data,
                typically loaded from the user's JSON file. This could be a dict
                if the JSON root is an object, or a list if it's a list of messages.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the extracted requirements
                                      in the structured format, or None if an error occurs.
        """
        if not self.model:
            print(
                "RequirementExtractor: Gemini model not initialized. Cannot extract requirements.")
            return None

        try:
            # Convert the conversation data to a JSON string to include in the prompt
            conversation_json_string = json.dumps(conversation_data, indent=2)
        except TypeError as e:
            print(f"Error converting conversation data to JSON string: {e}")
            return None

        prompt = self._create_extraction_prompt(conversation_json_string)

        print(
            "RequirementExtractor: Sending request to Gemini for requirement extraction...")
        try:
            # Configure for JSON output if the model supports it directly,
            # otherwise, we'll parse the text response.
            # For newer models, you might use response_mime_type="application/json"
            # generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
            # response = self.model.generate_content(prompt, generation_config=generation_config)

            # For now, assume text response and parse JSON from it
            response = self.model.generate_content(prompt)

            # Debug: Print the raw response text
            # print("--- Gemini Raw Response Text ---")
            # print(response.text)
            # print("--------------------------------")

            # Extract and parse the JSON from the response
            # The model should be prompted to ONLY output JSON.
            # Need to strip potential markdown backticks if model adds them
            json_response_text = response.text.strip()
            if json_response_text.startswith("```json"):
                json_response_text = json_response_text[7:]
            if json_response_text.endswith("```"):
                json_response_text = json_response_text[:-3]

            extracted_json = json.loads(json_response_text.strip())
            print(
                "RequirementExtractor: Successfully extracted and parsed requirements from Gemini.")
            return extracted_json
        except json.JSONDecodeError as e:
            print(
                f"RequirementExtractor: Error decoding JSON from Gemini response: {e}")
            # Log part of the problematic text
            print(f"Problematic text: '{response.text[:500]}...'")
            return None
        except Exception as e:
            print(
                f"RequirementExtractor: An error occurred during Gemini API call or processing: {e}")
            # You might want to inspect response.prompt_feedback here if available
            if hasattr(response, 'prompt_feedback'):
                print(f"Prompt Feedback: {response.prompt_feedback}")
            return None


# --- Example Usage (can be run directly for testing this module) ---
if __name__ == '__main__':
    print("Testing RequirementExtractor...")

    # For direct testing, load config and .env
    import sys
    project_root_for_test = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, project_root_for_test)
    from src.config import Config

    cfg = Config()

    # Ensure API key is loaded (Config class constructor should handle .env)
    if not cfg.GOOGLE_API_KEY or "FALLBACK" in cfg.GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not found or is a placeholder. Please set it in .env for testing.")
        exit()

    # Initialize RequirementExtractor
    # Use a capable model for this complex extraction task
    # If GEMINI_REQUIREMENT_EXTRACTION_MODEL is defined in config, use it. Otherwise, fallback.
    model_to_use = getattr(cfg, "GEMINI_REQUIREMENT_EXTRACTION_MODEL",
                           "models/gemini-1.5-flash-latest")  # Default if not in config

    extractor = RequirementExtractor(
        model_name=model_to_use, api_key=cfg.GOOGLE_API_KEY)

    if not extractor.model:
        print("Failed to initialize extractor model. Exiting test.")
        exit()

    # Load one of your sample conversation JSON files
    # Replace with the actual path to your dda70214...json file
    # Ensure this path is correct relative to where you run the script OR use absolute path
    conversation_file_path = os.path.join(
        cfg.PROJECT_ROOT, "conversation.json")  # Example path

    if not os.path.exists(conversation_file_path):
        print(f"Test conversation file not found: {conversation_file_path}")
        print(
            "Please place your sample conversation JSON in the project root or update path.")
    else:
        try:
            with open(conversation_file_path, 'r') as f:
                sample_conversation_data = json.load(f)
            print(
                f"\nLoaded sample conversation from: {conversation_file_path}")

            extracted_reqs = extractor.extract_requirements(
                sample_conversation_data)

            if extracted_reqs:
                print("\n--- Extracted Requirements ---")
                print(json.dumps(extracted_reqs, indent=2))
                print("-----------------------------")
            else:
                print("\nFailed to extract requirements from the sample conversation.")
        except Exception as e:
            print(f"Error loading or processing sample conversation file: {e}")

    print("\nRequirementExtractor test finished.")
