# ArchitecturalRAGSystem/src/rag_pipeline/query_generator.py
from typing import Dict, Any, List, Optional
# import google.generativeai as genai # Could use an LLM for advanced query generation
# from src.config import Config # If using LLM for query generation


class QueryGenerator:
    """
    Generates search queries for the RAG vector store based on
    structured user requirements.
    """

    def __init__(self, use_llm_for_generation: bool = False, llm_model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initializes the QueryGenerator.

        Args:
            use_llm_for_generation (bool): If True, use an LLM for more sophisticated query generation.
                                           If False, use rule-based/template-based generation.
            llm_model_name (Optional[str]): The Gemini model name if using LLM.
            api_key (Optional[str]): API key if using LLM.
        """
        self.use_llm_for_generation = use_llm_for_generation
        self.llm_model = None
        if self.use_llm_for_generation:
            if not llm_model_name:
                raise ValueError(
                    "llm_model_name must be provided if use_llm_for_generation is True.")
            # if api_key: # Configure genai if key is passed
            #     genai.configure(api_key=api_key)
            # try:
            #     self.llm_model = genai.GenerativeModel(llm_model_name)
            #     print(f"QueryGenerator: Initialized LLM model '{llm_model_name}' for query generation.")
            # except Exception as e:
            #     print(f"QueryGenerator: Error initializing LLM model '{llm_model_name}': {e}")
            #     self.use_llm_for_generation = False # Fallback to rule-based
            print("QueryGenerator: LLM-based query generation is not fully implemented in this basic version. Using rule-based.")
            self.use_llm_for_generation = False  # For now, stick to rule-based

    def _generate_rule_based_queries(self, extracted_requirements: Dict[str, Any]) -> List[str]:
        """
        Generates queries using predefined rules and templates based on extracted requirements.
        """
        queries = []

        # General Project Queries
        project_summary = extracted_requirements.get("project_summary", {})
        building_type = project_summary.get("building_type", "building")
        style = project_summary.get("user_style_preference", "")

        queries.append(
            f"General design standards for a {style} {building_type}")
        if project_summary.get("total_footprint_sqft"):
            queries.append(
                f"Space planning considerations for a {project_summary['total_footprint_sqft']} sq ft {building_type}")

        for constraint in project_summary.get("key_constraints_or_desires", []):
            queries.append(
                f"Standards or solutions for: {constraint} in a {building_type}")

        # Room Specific Queries
        for room_spec in extracted_requirements.get("room_specifications", []):
            room_name = room_spec.get("room_name", "room")
            room_attributes = room_spec.get("attributes", [])

            queries.append(
                f"Standard dimensions and layout for a {style} {room_name}")
            queries.append(f"Functional requirements for a {room_name}")

            if "kitchen" in room_name.lower():
                queries.append(
                    f"Electrical standards for a residential kitchen")
                queries.append(f"Plumbing standards for a residential kitchen")
                queries.append(f"Ventilation standards for a kitchen")
            elif "bath" in room_name.lower() or "washroom" in room_name.lower():
                queries.append(f"Plumbing standards for a bathroom")
                queries.append(f"Electrical safety standards for bathrooms")
                # If relevant
                queries.append(f"Accessibility standards for bathrooms")
            elif "bedroom" in room_name.lower():
                queries.append(f"Lighting standards for a bedroom")

            for attr in room_attributes:
                queries.append(
                    f"Design considerations for a {room_name} with attribute: {attr}")
                if "godfather vibes" in attr.lower() or "mafia style" in attr.lower():  # Example specific handling
                    queries.append(
                        f"Interior design elements for '{attr}' in an office or library")

        # Special Feature Queries
        for feature in extracted_requirements.get("special_features", []):
            feature_name = feature.get("feature_name")
            feature_desc = feature.get("description")
            if feature_name:
                queries.append(
                    f"Standards or examples for implementing: {feature_name}")
                if feature_desc:
                    # Use a snippet
                    queries.append(f"Design details for: {feature_desc[:100]}")

        # Site & Orientation Queries
        site_info = extracted_requirements.get("site_and_orientation", {})
        if site_info.get("lot_orientation_street_facing"):
            queries.append(
                f"Sunlight and passive design strategies for a {site_info['lot_orientation_street_facing']} facing lot")

        # Deduplicate queries (simple method)
        unique_queries = sorted(list(set(queries)))
        return unique_queries

    def _generate_llm_based_queries(self, extracted_requirements: Dict[str, Any]) -> List[str]:
        """
        (Placeholder) Uses an LLM to generate more nuanced queries.
        """
        # This would involve creating a prompt for Gemini, sending the extracted_requirements,
        # and asking it to generate a list of diverse and relevant search queries
        # for a knowledge base of architectural standards.
        print("LLM-based query generation called, but using rule-based fallback for now.")
        # Fallback to rule-based if LLM part is not implemented
        return self._generate_rule_based_queries(extracted_requirements)

    def generate_queries(self, extracted_requirements: Dict[str, Any]) -> List[str]:
        """
        Generates a list of search queries based on the structured requirements.

        Args:
            extracted_requirements (Dict[str, Any]): The structured output from
                                                     RequirementExtractor.

        Returns:
            List[str]: A list of string queries to be used for RAG retrieval.
        """
        if not extracted_requirements:
            print("QueryGenerator: No requirements provided, cannot generate queries.")
            return []

        print("QueryGenerator: Generating queries...")
        if self.use_llm_for_generation and self.llm_model:
            queries = self._generate_llm_based_queries(extracted_requirements)
        else:
            queries = self._generate_rule_based_queries(extracted_requirements)

        print(f"QueryGenerator: Generated {len(queries)} queries.")
        return queries


# --- Example Usage (can be run directly for testing this module) ---
if __name__ == '__main__':
    print("Testing QueryGenerator...")

    # For direct testing, we need to load a sample of extracted requirements
    # In a real pipeline, this would come from RequirementExtractor
    sample_extracted_requirements = {
        "project_summary": {
            "building_type": "House",
            "total_footprint_sqft": 1500,
            "num_floors": 3,
            "num_basements": 1,
            "user_style_preference": "Modern, Sophisticated",
            "budget_level": "Not specified",
            "key_constraints_or_desires": [
                "Total privacy for backyard",
                "Space for future elevator",
                "Hidden basement entrance"
            ]
        },
        "room_specifications": [
            {
                "room_name": "Bedroom",
                "quantity": 10,
                "attributes": [
                    "attached bathrooms"
                ],
                "connectivity_notes": []
            },
            {
                "room_name": "Drawing Room",
                "quantity": 1,
                "attributes": [
                    "washroom"
                ],
                "connectivity_notes": []
            },
            {
                "room_name": "Kitchen",
                "quantity": 1,
                "attributes": [
                    "large",
                    "closed",
                    "first floor",
                    "integrated dining area"
                ],
                "connectivity_notes": [
                    "connected to central living area"
                ]
            },
            {
                "room_name": "Living Area",
                "quantity": 2,
                "attributes": [
                    "central",
                    "functional space for TV and relaxing",
                    "balcony"
                ],
                "connectivity_notes": [
                    "connects to bedrooms and kitchen",
                    "access to backyard"
                ]
            },
            {
                "room_name": "Kids' Room",
                "quantity": 1,
                "attributes": [
                    "playroom"
                ],
                "connectivity_notes": [
                    "connected to central living area"
                ]
            },
            {
                "room_name": "Storage Room",
                "quantity": 2,
                "attributes": [
                    "separate",
                    "one on first floor",
                    "one large on rooftop"
                ],
                "connectivity_notes": []
            },
            {
                "room_name": "Gym",
                "quantity": 1,
                "attributes": [],
                "connectivity_notes": [
                    "open to basement",
                    "near gaming room",
                    "gradual flow to gaming room"
                ]
            },
            {
                "room_name": "Library",
                "quantity": 1,
                "attributes": [
                    "sophisticated",
                    "Godfather vibes"
                ],
                "connectivity_notes": [
                    "open to basement",
                    "near office"
                ]
            },
            {
                "room_name": "Gaming Room",
                "quantity": 1,
                "attributes": [
                    "modern sci-fi vibes"
                ],
                "connectivity_notes": [
                    "open to basement",
                    "near gym",
                    "gradual flow from gym"
                ]
            },
            {
                "room_name": "Office",
                "quantity": 1,
                "attributes": [
                    "organized",
                    "minimal",
                    "Italian/Mafia style",
                    "Godfather vibes"
                ],
                "connectivity_notes": [
                    "open to basement",
                    "near library"
                ]
            },
            {
                "room_name": "Chill and Relax Area",
                "quantity": 1,
                "attributes": [
                    "near gym",
                    "comfortable seating",
                    "soft lighting",
                    "natural elements",
                    "entertainment",
                    "refreshments"
                ],
                "connectivity_notes": [
                    "open to basement"
                ]
            },
            {
                "room_name": "Washroom",
                "quantity": 3,
                "attributes": [],
                "connectivity_notes": [
                    "one accessible from drawing room",
                    "one on rooftop"
                ]
            },
            {
                "room_name": "Powder Room",
                "quantity": 1,
                "attributes": [
                    "if space available"
                ],
                "connectivity_notes": []
            },
            {
                "room_name": "Outdoor Kitchen",
                "quantity": 2,
                "attributes": [
                    "portable/semi-portable",
                    "dedicated area"
                ],
                "connectivity_notes": [
                    "ground floor",
                    "rooftop"
                ]
            }
        ],
        "special_features": [
            {
                "feature_name": "Hidden Basement Access",
                "description": "hidden in a wall using an optical illusion, abstract 3D sculpture, standard door size with organic shape",
                "related_rooms": [
                    "Basement",
                    "Living Area"
                ]
            },
            {
                "feature_name": "Backyard",
                "description": "mini-garden, pool, bonfire place, tea-sitting area, swings for kids, totally private",
                "related_rooms": [
                    "Living Area"
                ]
            },
            {
                "feature_name": "Rooftop",
                "description": "sitting area, party area, washroom, big storage room, plants (flowers for pictures), lighting, outdoor kitchen area",
                "related_rooms": []
            }
        ],
        "site_and_orientation": {
            "lot_shape": "Rectangular",
            "lot_orientation_street_facing": "North is to the left when facing the street",
            "lot_width_vs_depth": "Deeper than wide (longer side faces street)"
        }
    }

    # Test with rule-based generation first
    rule_based_generator = QueryGenerator(use_llm_for_generation=False)
    generated_queries_rule = rule_based_generator.generate_queries(
        sample_extracted_requirements)

    print("\n--- Rule-Based Generated Queries ---")
    if generated_queries_rule:
        # Print first 10
        for i, q_text in enumerate(generated_queries_rule[:10]):
            print(f"  {i+1}. {q_text}")
        if len(generated_queries_rule) > 10:
            print(
                f"  ... and {len(generated_queries_rule) - 10} more queries.")
    else:
        print("  No queries generated.")

    # Placeholder for testing LLM-based generation (currently falls back to rule-based)
    # import sys, os
    # project_root_for_test = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # sys.path.insert(0, project_root_for_test)
    # from src.config import Config
    # cfg = Config()
    # if cfg.GOOGLE_API_KEY and "FALLBACK" not in cfg.GOOGLE_API_KEY:
    #     llm_generator = QueryGenerator(
    #         use_llm_for_generation=True, # Set to True to test LLM path (currently placeholder)
    #         llm_model_name=getattr(cfg, "GEMINI_QUERY_GENERATION_MODEL", "models/gemini-1.5-flash-latest"),
    #         api_key=cfg.GOOGLE_API_KEY
    #     )
    #     generated_queries_llm = llm_generator.generate_queries(sample_extracted_requirements)
    #     print("\n--- LLM-Based Generated Queries (currently placeholder) ---")
    #     if generated_queries_llm:
    #         for i, q_text in enumerate(generated_queries_llm[:10]):
    #             print(f"  {i+1}. {q_text}")
    #     else:
    #         print("  No queries generated by LLM (or fallback).")
    # else:
    #     print("\nSkipping LLM-based query generation test: API key not configured or placeholder.")

    print("\nQueryGenerator test finished.")
