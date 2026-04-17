from datetime import datetime
from Activity import Activity
from Activity_Analyzer import Activity_Analyzer
from Activity_Manager import Activity_Manager
from category_classifier import classify_activity

# Convert input time into a datetime object so program can  calculate
def parse_time(t):
    try:
        return datetime.strptime(t.strip(), "%I:%M %p")
    except ValueError:
        try:
            return datetime.strptime(t.strip(), "%I %p")
        except ValueError:
            # If both fail return None so caller knows it's invalid
            return None

def main():
    manager = Activity_Manager()
    analyzer = Activity_Analyzer(manager)


    print("Welcome to your WeekFrame...\n--------------------------")
    while True:
        choice = input("What do you wanna do?\nLog activity (enter 'log')\nSee insights (enter 'insights'): ")
        if choice.lower() == "log":
            while True:
                start_time_input = input("Start time (ex 3:20 PM): ")
                start_time = parse_time(start_time_input)
                if start_time is None:
                    print("Invalid time format. Use something like 3:20 PM or 11 PM.")
                else:
                    break

            while True:
                end_time_input = input("End time (ex 4:10 PM): ")
                end_time = parse_time(end_time_input)
                if end_time is None:
                    print("Invalid time format. Use something like 4:10 PM or 9 PM.")
                else:
                    break
                
            activity_note = input("Tell more about what you did? ").strip()

            # notes cannot be empty
            while not activity_note:
                activity_note = input("Note cannot be empty. Tell more about what you did? ").strip()

            predicted_category = classify_activity(activity_note)
            category = input(f"Suggested category: {predicted_category} (press Enter to accept or type your own): ").strip()

            # If user leaves it empty use the prediction
            if not category:
                category = predicted_category
                
            notes = activity_note


            # Convert datetime object back to  AM/PM format
            start_time_str = start_time.strftime("%I:%M %p")
            end_time_str = end_time.strftime("%I:%M %p")

            manager.add_activity(category, start_time_str, end_time_str, notes)
            manager.write_data()

        elif choice.lower() == "insights":
            print("Insights on your week:")
            insights_available = analyzer.weekly_insights()
            if insights_available:
                week_compare_choice = input("Do you wanna compare to last week progress? (Yes/No): ")
                if week_compare_choice.lower() == "yes":
                    print("Comparing...")
                    analyzer.compare_to_last_week()
                elif week_compare_choice.lower() == "no":
                    print("Exiting insights.")
                else:
                    print("Invalid choice. Please enter Yes or No.")
            else:
                print("Not enough data for insights.")

        else:
            print("Error: Enter 'log' or 'insights'.")

main()
