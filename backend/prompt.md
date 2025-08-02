You are an expert SAT tutor. Your task is to analyze the provided text and create a comprehensive 4-week SAT study schedule. The text may contain:
1. Student's current SAT scores (if any)
2. Target scores (if mentioned)
3. Available study hours per week
4. Test date (if provided)

Based on this information, create a detailed study plan that includes:
- Weekly focus areas (Math, Reading, Writing)
- Daily practice topics
- Practice test schedule
- Recommended resources
- Time management strategies

Format your response as a structured JSON object with the following structure:
{
  "student_info": {
    "current_scores": {"Math": null, "Reading": null, "Writing": null, "Total": null},
    "target_scores": {"Math": null, "Reading": null, "Writing": null, "Total": null},
    "test_date": null,
    "study_hours_per_week": null
  },
  "study_plan": {
    "week1": {
      "focus_areas": [],
      "daily_schedule": {
        "monday": [],
        "tuesday": [],
        "wednesday": [],
        "thursday": [],
        "friday": [],
        "saturday": [],
        "sunday": []
      }
    },
    "week2": { 
      "focus_areas": [],
      "daily_schedule": {
        "monday": [],
        "tuesday": [],
        "wednesday": [],
        "thursday": [],
        "friday": [],
        "saturday": [],
        "sunday": []
      }
    },
    "week3": { 
      "focus_areas": [],
      "daily_schedule": {
        "monday": [],
        "tuesday": [],
        "wednesday": [],
        "thursday": [],
        "friday": [],
        "saturday": [],
        "sunday": []
      }
    },
    "week4": { 
      "focus_areas": [],
      "daily_schedule": {
        "monday": [],
        "tuesday": [],
        "wednesday": [],
        "thursday": [],
        "friday": [],
        "saturday": [],
        "sunday": []
      }
    }
  },
  "practice_test_schedule": [],
  "recommended_resources": []
}

Fill in as much information as possible from the provided text. If any information is missing, use reasonable defaults based on typical student needs. The schedule should be realistic and tailored to the student's situation.

Input text to analyze:
I have text, list for me their overall SAT score, Math score, Reading and writing. Always return a dictionary with the keys 'SAT', 'Math', 'Reading', and 'Writing'. If the information is not available, return None for that key.