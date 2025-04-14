# app/core/feedback_reconciliation.py
from typing import Dict, List, Tuple, Any
import re
from app.utils.garment_detection import detect_potential_dress, get_dress_type_from_items

class FeedbackReconciler:
    """
    Ensures consistency between detected items and generated feedback
    by validating and reconciling any discrepancies
    """
    
    def __init__(self):
        # Define mapping between common mismatches
        self.garment_mapping = {
            "wide_leg_jeans": "pants",
            "slacks": "pants",
            "trousers": "pants",
            "wide_leg_trousers": "pants",
            "maxi_skirt": "skirt",
            "midi_skirt": "skirt",
            "mini_skirt": "skirt",
        }
        
        # Common replacements in feedback to ensure consistency
        self.feedback_replacements = {
            "maxi skirt": "pants",
            "skater skirt": "skirt",
            "pleated skirt": "skirt"
        }
        
    def reconcile_results(self, result: Dict[str, Any], clothing_items: List[Dict] = None, color_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reconcile all components of the results to ensure consistency
        
        Args:
            result: The complete analysis result
            clothing_items: Optional list of detected clothing items
            color_analysis: Optional color analysis results
            
        Returns:
            dict: The reconciled result
        """
        # Create a working copy of the result
        reconciled = result.copy()
        
        # Extract key information
        labels = reconciled.get("labels", {})
        feedback = reconciled.get("feedback", {})
        clothing_items = clothing_items or reconciled.get("clothing_items", [])
        color_analysis = color_analysis or reconciled.get("color_analysis", {})
        
        # 1. Reconcile garment types
        self._reconcile_garment_types(labels, clothing_items, color_analysis, feedback)
        
        # 2. Reconcile accessories
        self._reconcile_accessories(labels, clothing_items, feedback)
        
        # 3. Reconcile footwear
        self._reconcile_footwear(labels, clothing_items, feedback)
        
        # 4. Reconcile feedback text with labels
        self._reconcile_feedback_text(labels, feedback)
        
        # 5. Reconcile style feedback
        self._reconcile_style_feedback(labels, feedback)
        
        # Return the reconciled result
        return reconciled
    
    def _reconcile_garment_types(self, labels: Dict[str, Any], clothing_items: List[Dict], 
                                color_analysis: Dict[str, Any], feedback: Dict[str, str]):
        """
        Ensure consistency in garment type naming with improved dress detection
        """
        # Check for potential dress
        is_likely_dress = detect_potential_dress(clothing_items, color_analysis)
        
        # If we detected a likely dress but it's labeled as separate pieces
        if is_likely_dress and (
            "skirt" in labels.get("pants_type", "").lower() or
            labels.get("top_type", "") == "unknown_top" or
            (labels.get("top_type", "") and labels.get("pants_type", "") and 
             any("top" in item["category"] for item in clothing_items) and 
             any("bottom" in item["category"] for item in clothing_items))
        ):
            # Get a proper dress type
            dress_type = get_dress_type_from_items(clothing_items, color_analysis)
            
            # Update labels to correctly identify as a dress
            labels["top_type"] = dress_type
            labels["pants_type"] = "none"  # No separate bottom for dresses
            
            # Also update feedback to reflect the dress
            for key in feedback:
                # Replace "top and skirt" or similar phrases with "dress"
                feedback[key] = feedback[key].replace("top and skirt", "dress")
                feedback[key] = feedback[key].replace("top with skirt", "dress")
                feedback[key] = feedback[key].replace("layering", "dress design")
                # For any other combinations of top + bottom terms
                bottom_terms = ["skirt", "bottom", "pants"]
                top_terms = ["top", "blouse", "shirt"]
                
                for bottom in bottom_terms:
                    for top in top_terms:
                        combined = f"{top} and {bottom}"
                        feedback[key] = feedback[key].replace(combined, "dress")
                        combined = f"{top} with {bottom}"
                        feedback[key] = feedback[key].replace(combined, "dress")
            
            return
        
        # Get information about bottom garments
        bottom_items = [item for item in clothing_items if item.get("category") == "bottom"]
        
        # If pants_type contains "skirt" but we have bottom items that are not skirts
        if "skirt" in labels.get("pants_type", "").lower():
            # Check if we have detected pants
            pants_detected = any("pant" in item.get("type", "").lower() or 
                              "trouser" in item.get("type", "").lower() or 
                              "jean" in item.get("type", "").lower() 
                              for item in bottom_items)
            
            if pants_detected:
                # Replace skirt with pants in the pants_type
                if "wide" in labels.get("pants_type", "").lower():
                    labels["pants_type"] = "wide_leg_pants"
                else:
                    labels["pants_type"] = "pants"
        
        # Conversely, if pants_type says "pants" but we have skirts
        elif "pant" in labels.get("pants_type", "").lower():
            # Check if we have detected skirts
            skirt_detected = any("skirt" in item.get("type", "").lower() for item in bottom_items)
            
            if skirt_detected and not any("pant" in item.get("type", "").lower() or 
                                      "trouser" in item.get("type", "").lower() or 
                                      "jean" in item.get("type", "").lower() 
                                      for item in bottom_items):
                # Replace pants with skirt
                if "wide" in labels.get("pants_type", "").lower() or "maxi" in labels.get("pants_type", "").lower():
                    labels["pants_type"] = "maxi_skirt"
                else:
                    labels["pants_type"] = "skirt"
    
    def _reconcile_accessories(self, labels: Dict[str, Any], clothing_items: List[Dict], feedback: Dict[str, str]):
        """
        Ensure consistency between detected accessories and feedback
        """
        # Check if accessories were detected
        detected_accessories = labels.get("accessories", [])
        
        # If feedback says there are no accessories but we have detected some
        if "missing accessories" in feedback.get("accessories", "").lower() and detected_accessories:
            # Update feedback to mention the accessories we found
            if len(detected_accessories) == 1:
                acc_name = detected_accessories[0]
                feedback["accessories"] = f"The {acc_name} adds a nice touch to the outfit. Consider adding more accessories like a statement necklace or earrings for additional visual interest."
            else:
                acc_list = ", ".join(detected_accessories[:-1]) + " and " + detected_accessories[-1]
                feedback["accessories"] = f"The {acc_list} complement the outfit well. They add visual interest and complete the look."
        
        # If feedback mentions accessories but we didn't detect any
        elif detected_accessories and "completing the look" not in feedback.get("accessories", "").lower():
            # Update the feedback to acknowledge the accessories
            acc_string = ", ".join(detected_accessories)
            feedback["accessories"] = f"The {acc_string} add(s) a nice touch to the outfit. They complement the overall style well."
    
    def _reconcile_footwear(self, labels: Dict[str, Any], clothing_items: List[Dict], feedback: Dict[str, str]):
        """
        Ensure consistency between detected footwear and feedback
        """
        # Check if footwear is visible in the image
        footwear_type = labels.get("footwear_type", "")
        
        # Look for any specific footwear mentions in the feedback
        footwear_feedback = feedback.get("footwear", "")
        
        # If footwear is marked as not visible, but we can see it's visible in the image
        is_footwear_visible = footwear_type != "footwear_not_visible"
        
        # If we can see footwear but feedback says otherwise
        if is_footwear_visible and "consider adding" in footwear_feedback.lower():
            # Update feedback to acknowledge the visible footwear
            formatted_type = footwear_type.replace("_", " ")
            feedback["footwear"] = f"The {formatted_type} complement the outfit well. They add a practical yet stylish element to the look."
        
        # If we can't see footwear but feedback talks about it
        elif not is_footwear_visible and "complement" in footwear_feedback.lower():
            # Update feedback to note that footwear isn't visible
            feedback["footwear"] = "Consider adding appropriate footwear to complete the look."
    
    def _reconcile_feedback_text(self, labels: Dict[str, Any], feedback: Dict[str, str]):
        """
        Ensure feedback text matches labels by replacing inconsistent terms
        """
        # Check if this is a dress
        is_dress = (
            "dress" in labels.get("top_type", "").lower() or
            "tunic" in labels.get("top_type", "").lower() or
            labels.get("pants_type", "") == "none"
        )
        
        if is_dress:
            # Replace any mentions of "top and bottom" with "dress"
            for component in feedback:
                feedback[component] = feedback[component].replace("top and bottom", "dress")
                feedback[component] = feedback[component].replace("top works well with the bottom", "dress creates a cohesive look")
                feedback[component] = feedback[component].replace("proportion", "silhouette")
                feedback[component] = feedback[component].replace("proportions", "silhouette")
                
        # Check fit feedback
        fit_feedback = feedback.get("fit", "")
        pants_type = labels.get("pants_type", "")
        
        # Replace any mentions of skirt with pants or vice versa
        for incorrect, correct in self.feedback_replacements.items():
            if incorrect in fit_feedback.lower() and correct in pants_type.lower():
                feedback["fit"] = fit_feedback.lower().replace(incorrect, correct)
            
        # Make similar replacements for other feedback sections
        for section in ["color", "style"]:
            section_feedback = feedback.get(section, "")
            for incorrect, correct in self.feedback_replacements.items():
                if incorrect in section_feedback.lower() and correct in pants_type.lower():
                    feedback[section] = section_feedback.lower().replace(incorrect, correct)
    
    def _reconcile_style_feedback(self, labels: Dict[str, Any], feedback: Dict[str, str]):
        """
        Ensure style feedback matches detected style
        """
        style = labels.get("style", "").replace("_", " ")
        
        # If the feedback mentions a different style than what was detected
        if style and "style" in feedback:
            style_pattern = r'of ([a-zA-Z_\s]+) style'
            match = re.search(style_pattern, feedback["style"])
            if match and match.group(1).lower() != style.lower():
                # Replace the incorrect style with the detected one
                feedback["style"] = feedback["style"].replace(
                    match.group(1), style
                )


def reconcile_fashion_analysis(result: Dict[str, Any], clothing_items: List[Dict] = None, color_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to reconcile fashion analysis results
    
    Args:
        result: Complete fashion analysis result
        clothing_items: Optional list of detected clothing items
        color_analysis: Optional color analysis results
        
    Returns:
        dict: Reconciled fashion analysis
    """
    reconciler = FeedbackReconciler()
    return reconciler.reconcile_results(result, clothing_items, color_analysis)