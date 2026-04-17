import pandas as pd

class Activity_Analyzer:
    def __init__(self, activity_manager):
        self.activity_manager = activity_manager

    def get_data(self):
        return self.activity_manager.get_data()

    def prepare_data(self) -> pd.DataFrame:
        
        # Converts raw activity data into a pandas DataFrame
        
        raw_data = self.get_data()
        if not raw_data:
            return pd.DataFrame()

        df = pd.DataFrame(raw_data)

        # Convert string dates and times to datetime objects
        df['start_time'] = pd.to_datetime(df['start_time'], format="%I:%M %p")
        df['end_time'] = pd.to_datetime(df['end_time'], format="%I:%M %p")
        df['date'] = pd.to_datetime(df['date'])

        # Calculate duration in hours -overnight activities
        durations = []
        for _, row in df.iterrows():
            start = row['start_time']
            end = row['end_time']
            if end < start:
                end += pd.Timedelta(days=1)
            durations.append((end - start).total_seconds() / 3600)
        df['duration_hours'] = durations

        # Normalize category names to avoid mismatch issues
        df['category'] = df['category'].str.strip().str.lower()

        # Tag each row with the week start date
        df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

        return df

    def weekly_insights(self):
        """
        Prints weekly insights for the last week of data
        Returns True if insights were printed
        """
        df = self.prepare_data()
        if df.empty:
            print("No data logged yet.")
            return False

        # Get all categories
        all_categories = ['exercise', 'eating', 'leisure-positive','leisure-negative', 'study', 'sleep', 'social','self-care']

        # Determine last week
        weeks = sorted(df['week'].unique())
        if not weeks:
            print("Not enough data for weekly insights.")
            return False

        current_week = weeks[-1]
        df_week = df[df['week'] == current_week]

        # Group by category and sum durations ensure all categories are included
        total_hours = df_week.groupby('category')['duration_hours'] \
                            .sum().reindex(all_categories, fill_value=0).round(2)

        # Calculate percentages of total logged time
        total_time = total_hours.sum()
        percentages = (total_hours / total_time * 100).round(2) if total_time > 0 else total_hours*0

        # Most and least time spent categories
        most_category = total_hours.idxmax()
        least_category = total_hours.idxmin()

        print("\nThis week's insights:")
        print(f"Most time spent on: {most_category} - {total_hours[most_category]:.2f} hours")
        print(f"Least time spent on: {least_category} - {total_hours[least_category]:.2f} hours")

        # Time by category with percentages
        print("\nTime by category:")
        for cat in all_categories:
            print(f"  {cat}: {total_hours[cat]:.2f} hours ({percentages[cat]:.2f}% of total week)")

        # Identify days with low activity logging (<5 hours)
        daily_hours = df_week.groupby('date')['duration_hours'].sum()
        low_activity_days = daily_hours[daily_hours < 5]
        if not low_activity_days.empty:
            print("\nDays with low activity logging (<5 hours):")
            for date, hours in low_activity_days.items():
                print(f"  {date.date()}: {hours:.2f} hours")

        return True

    def compare_to_last_week(self):
        """
        Compares total hours per category between the last two weeks
        Prints differences clearly
        """
        df = self.prepare_data()
        if df.empty:
            print("No data available.")
            return

        all_categories = ['exercise', 'eating', 'leisure-positive','leisure-negative', 'study', 'sleep', 'social','self-care']

        weeks = sorted(df['week'].unique())
        if len(weeks) < 2:
            print("Not enough data to compare weeks.")
            return

        last_week_start = weeks[-2]
        current_week_start = weeks[-1]

        df_last = df[df['week'] == last_week_start]
        df_current = df[df['week'] == current_week_start]

        last_hours = df_last.groupby('category')['duration_hours'].sum().reindex(all_categories, fill_value=0).round(2)
        current_hours = df_current.groupby('category')['duration_hours'].sum().reindex(all_categories, fill_value=0).round(2)

        print("\nComparing to last week:")
        for cat in all_categories:
            curr = current_hours[cat]
            prev = last_hours[cat]
            diff = curr - prev

            if diff > 0:
                print(f"{cat}: {curr:.2f} hours - {diff:.2f} more than last week")
            elif diff < 0:
                print(f"{cat}: {curr:.2f} hours - {abs(diff):.2f} hours less than last week")
            else:
                print(f"{cat}: {curr:.2f} hours - same as last week")

            # Optional health suggestion if drop is significant (>50% drop)
            if prev > 0 and diff/prev < -0.5:
                print(f"  Note: Significant drop in {cat} this week. Consider adjusting your routine.")

