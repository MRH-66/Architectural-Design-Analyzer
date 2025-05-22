# ArchitecturalRAGSystem/src/rag_pipeline/synthesizer.py
import google.generativeai as genai
import json
from typing import Dict, Any, Optional, List
import time
import os

# from src.config import Config # Will be used when called


class Synthesizer:
    """
    Synthesizes user requirements and RAG-retrieved context into a final
    detailed JSON output, including specific standards and BOQ information.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initializes the Synthesizer.

        Args:
            model_name (str): The name of the powerful Gemini model to use for synthesis
                              (e.g., "models/gemini-2.5-flash-preview-04-17").
            api_key (Optional[str]): The Google API Key. If None, assumes
                                     genai.configure() has been called.
        """
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)

        try:
            self.model = genai.GenerativeModel(self.model_name)
            print(
                f"Synthesizer: Initialized Gemini model '{self.model_name}'.")
        except Exception as e:
            print(
                f"Error initializing Gemini model '{self.model_name}' for Synthesizer: {e}")
            self.model = None

    def _construct_synthesis_prompt(self,
                                    extracted_requirements: Dict[str, Any],
                                    retrieved_contexts_per_query: Dict[str, List[Dict[str, Any]]],
                                    output_json_schema_example: Optional[str] = None
                                    ) -> str:
        """
        Constructs the detailed prompt for Gemini to synthesize the final output.
        This is the MOST CRITICAL prompt and will require extensive iteration.
        """

        # Convert contexts to a more readable string format for the prompt
        context_str_parts = []
        for query, contexts in retrieved_contexts_per_query.items():
            if contexts:  # Only include if there's context for the query
                context_str_parts.append(
                    f"For the query '{query}', relevant information from architectural standards includes:")
                for i, ctx in enumerate(contexts):
                    text = ctx.get('text', 'N/A')
                    metadata = ctx.get('metadata', {})
                    source = metadata.get('source_document', 'Unknown source')
                    page = metadata.get('original_page_number', 'N/A')
                    context_str_parts.append(
                        # Show a snippet
                        f"  Context {i+1} (Source: {source}, Page: {page}):\n    \"{text[:500].strip()}...\"")
                context_str_parts.append("-" * 20)

        full_context_str = "\n".join(context_str_parts)
        if not full_context_str:
            full_context_str = "No specific context was retrieved from the knowledge base for the generated queries. Please rely on general architectural knowledge and the user's requirements."

        # Define a more detailed desired output schema within the prompt
        # This is a simplified example; your actual schema will be more complex for BOQs
        # You might want to load this schema from a separate JSON file for maintainability.
        if not output_json_schema_example:
            output_json_schema_example = """
            {
              "project_summary_assessment": {
                "building_type": "User's building_type",
                "style_assessment": "Brief assessment of applying user's style (e.g., 'Modern, Sophisticated')",
                "footprint_sqft": "User's total_footprint_sqft",
                "num_floors": "User's num_floors",
                "num_basements": "User's num_basements",
                "overall_design_notes": ["Key design driver 1 from user requirements", "Key design driver 2 based on standards and user needs"]
              },
              "room_detailed_standards": [
                {
                  "room_name": "Example: Master Bedroom",
                  "user_attributes_and_connectivity": ["User attribute 1", "User connectivity note 1"],
                  "derived_dimensions_ft": { // Attempt to derive or suggest based on context or general standards
                    "length": "e.g., 14-16 or Standard X", 
                    "width": "e.g., 12-14 or Standard Y",
                    "min_ceiling_height_ft": "e.g., 9 or Standard Z (Source: Book A, p.XX)"
                  },
                  "electrical_boq_items": [
                    {"item": "Ceiling Light Point", "quantity": 1, "standard_source": "Neufert p.X / General Practice"},
                    {"item": "General Purpose Outlet (Double)", "quantity": 4, "standard_source": "General Residential Code / Book B p.Y"},
                    {"item": "Bedside Outlet (Double)", "quantity": 2}
                  ],
                  "plumbing_boq_items_associated_bath": [ // Only if it has an attached bath
                    {"item": "WC", "quantity": 1, "standard_source": "Book C p.Z"},
                    {"item": "Wash Basin", "quantity": 1}, // Or 2 if master
                    {"item": "Shower Point", "quantity": 1}
                  ],
                  "hvac_notes": ["Standard ventilation requirements", "Consider individual thermostat if requested"],
                  "finishes_style_notes": ["Align with 'Modern, Sophisticated'. E.g., neutral palette, feature wall material suggestion."]
                }
                // ... more rooms ...
              ],
              "civil_boq_general_standards": {
                "standard_internal_wall_thickness_inches": "e.g., 4.5 or 6 (Source: Book D, p.A)",
                "standard_external_wall_thickness_inches": "e.g., 9 or 12 (Source: Book D, p.B)",
                "foundation_type_suggestion": "Based on X stories and soil (if mentioned, else general type)"
              },
              "general_electrical_notes": ["Overall panel sizing considerations (e.g., 200A service for this size house)"],
              "general_plumbing_notes": ["Hot water system type suggestion"],
              "unresolved_or_conflicting_requirements": [
                  {"issue": "e.g., User wants 10 bedrooms in 1500 sq ft total, which is not feasible per standard room sizes.", "suggestion": "Clarify with user or prioritize essential rooms."}
              ],
              "warnings_and_disclaimers": ["All standards and quantities are preliminary and must be verified with local codes and professional engineers/architects."]
            }
            """

        prompt = f"""
        You are an expert AI Architectural Design Assistant. Your task is to synthesize detailed design parameters,
        standards, and preliminary Bill of Quantities (BOQ) information based on user requirements and
        retrieved architectural standards.

        **1. User's Extracted Requirements:**
        ```json
        {json.dumps(extracted_requirements, indent=2)}
        ```

        **2. Relevant Context from Architectural Standards Books (Retrieved by RAG):**
        ```
        {full_context_str}
        ```

        **Task:**
        Based *only* on the User's Extracted Requirements and the Relevant Context provided above, generate a comprehensive JSON output.
        Follow the schema and example structure provided below very carefully.

        **Detailed Instructions:**
        - For each room in "room_detailed_standards":
            - Refer to the user's attributes and connectivity for that room.
            - From the "Relevant Context", extract specific dimensional standards (length, width, ceiling height), number of electrical points (by type), and standard plumbing fixtures. Cite the source (book/page from context metadata if available) for critical standards.
            - If a specific standard is not found in the context for a user's request, state "Standard not found in provided context; general practice/local code verification needed" or apply a very common default and note it. For example, if no ceiling height is found, you might suggest 9ft for residential.
            - **Do NOT invent standards or numbers if not supported by the provided context or very common general knowledge you are explicitly asked to use as a fallback.**
            - For BOQ items, list the item and its quantity per room.
        - For "civil_boq_general_standards", provide general architectural norms for wall thicknesses, etc., if found in context or generally known.
        - In "unresolved_or_conflicting_requirements", list any major issues where user desires conflict with standard practices or available space (e.g., too many rooms for the footprint, if calculable).
        - The user's preference for "Modern, Sophisticated" style should influence notes on finishes or general design approach where applicable.
        - **Crucially, if the `extracted_requirements` (e.g., number of bedrooms, number of floors) seem implausible or conflict with common sense for the given footprint, highlight this in "unresolved_or_conflicting_requirements".** For example, 10 bedrooms with attached baths plus many other rooms in a 1500 sq ft footprint for the entire house is not feasible. If the footprint is per floor for a multi-story house, ensure calculations are based on that. The user's requirements provided above are the primary source for what they want.
        - The BOQs should list items and quantities/types, not costs.

        **Output Schema and Example Structure:**
        ```json
        {output_json_schema_example}
        ```

        **Output ONLY the fully populated JSON object adhering to this schema. Do not include any other explanatory text before or after the JSON.**
        """
        return prompt

    def synthesize_output(self,
                          extracted_requirements: Dict[str, Any],
                          retrieved_contexts_per_query: Dict[str,
                                                             List[Dict[str, Any]]]
                          ) -> Optional[Dict[str, Any]]:
        """
        Generates the final structured JSON output by synthesizing requirements and context.

        Args:
            extracted_requirements (Dict[str, Any]): Output from RequirementExtractor.
            retrieved_contexts_per_query (Dict[str, List[Dict[str, Any]]]): Output from RAG retrieval
                (query string -> list of retrieved chunk dicts).

        Returns:
            Optional[Dict[str, Any]]: The final synthesized JSON, or None if an error occurs.
        """
        if not self.model:
            print("Synthesizer: Gemini model not initialized. Cannot synthesize output.")
            return None

        prompt = self._construct_synthesis_prompt(
            extracted_requirements, retrieved_contexts_per_query)

        # For debugging the prompt:
        # print("\n--- SYNTHESIS PROMPT ---")
        # print(prompt[:2000]) # Print first 2000 chars
        # print("...")
        # print(prompt[-2000:]) # Print last 2000 chars
        # print("--- END SYNTHESIS PROMPT ---\n")

        print("Synthesizer: Sending request to Gemini for final synthesis...")
        start_time = time.time()
        try:
            # For models that can directly output JSON:
            # generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
            # response = self.model.generate_content(prompt, generation_config=generation_config)

            # Assuming text response for now, parse JSON from it
            response = self.model.generate_content(prompt)
            end_time = time.time()
            print(
                f"Synthesizer: Received response from Gemini in {end_time - start_time:.2f} seconds.")

            # Debug raw response
            # print("--- GEMINI SYNTHESIS RAW RESPONSE ---")
            # print(response.text)
            # print("--- END GEMINI SYNTHESIS RAW RESPONSE ---")

            json_response_text = response.text.strip()
            if json_response_text.startswith("```json"):
                json_response_text = json_response_text[7:]
            if json_response_text.endswith("```"):
                json_response_text = json_response_text[:-3]

            synthesized_json = json.loads(json_response_text.strip())
            print("Synthesizer: Successfully synthesized and parsed final JSON output.")
            return synthesized_json
        except json.JSONDecodeError as e:
            print(
                f"Synthesizer: Error decoding JSON from Gemini synthesis response: {e}")
            print(f"Problematic text snippet: '{response.text[:1000]}...'")
            return None
        except Exception as e:
            print(
                f"Synthesizer: An error occurred during Gemini API call or synthesis: {e}")
            if hasattr(response, 'prompt_feedback'):
                print(f"Prompt Feedback: {response.prompt_feedback}")
            return None


# --- Example Usage (can be run directly for testing this module) ---
if __name__ == '__main__':
    print("Testing Synthesizer...")

    import sys
    project_root_for_test = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, project_root_for_test)
    from src.config import Config

    cfg = Config()

    if not cfg.GOOGLE_API_KEY or "FALLBACK" in cfg.GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not found or is a placeholder. Please set it in .env for testing.")
        exit()

    # Use a powerful model for synthesis
    synthesis_model_name = getattr(
        cfg, "GEMINI_SYNTHESIS_MODEL", "models/gemini-2.5-flash-preview-04-17")
    synthesizer = Synthesizer(
        model_name=synthesis_model_name, api_key=cfg.GOOGLE_API_KEY)

    if not synthesizer.model:
        print("Failed to initialize synthesizer model. Exiting test.")
        exit()

    # Load sample intermediate RAG output (from run_query_service.py)
    # This file should contain "extracted_requirements" and "retrieved_contexts_per_query"
    intermediate_file_path = os.path.join(
        # Adjust if filename differs
        cfg.OUTPUT_JSON_PATH, "rag_intermediate_output_conversation.json")

    if not os.path.exists(intermediate_file_path):
        print(
            f"Intermediate RAG output file not found: {intermediate_file_path}")
        print("Please run 'run_query_service.py' first to generate this file or provide a sample.")
    else:
        try:
            with open(intermediate_file_path, 'r', encoding='utf-8') as f:
                intermediate_data = json.load(f)
            print(
                f"\nLoaded intermediate RAG data from: {intermediate_file_path}")

            extracted_reqs = intermediate_data.get("extracted_requirements")
            retrieved_ctx = intermediate_data.get(
                "retrieved_contexts_per_query")

            if extracted_reqs and retrieved_ctx is not None:  # retrieved_ctx can be an empty dict
                final_output_json = synthesizer.synthesize_output(
                    extracted_reqs, retrieved_ctx)

                if final_output_json:
                    print("\n--- Final Synthesized Output JSON ---")
                    print(json.dumps(final_output_json, indent=2))
                    print("------------------------------------")

                    # Optionally save this final output
                    final_output_filename = f"final_synthesized_design_{os.path.basename(intermediate_data.get('user_conversation_file', 'unknown_input')).split('.')[0]}.json"
                    final_output_filepath = os.path.join(
                        cfg.OUTPUT_JSON_PATH, final_output_filename)
                    with open(final_output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(final_output_json, f,
                                  indent=2, ensure_ascii=False)
                    print(
                        f"\nSaved final synthesized output to: {final_output_filepath}")
                else:
                    print("\nFailed to synthesize final output JSON.")
            else:
                print(
                    "Missing 'extracted_requirements' or 'retrieved_contexts_per_query' in the intermediate file.")
        except Exception as e:
            print(
                f"Error loading or processing intermediate RAG output file: {e}")

    print("\nSynthesizer test finished.")
