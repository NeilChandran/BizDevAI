import pandas as pd
import datetime

class KPIReporter:
    def __init__(self, lead_file='data/all_leads_cleaned.csv', out_file='reports/kpi_report.csv'):
        self.lead_file = lead_file
        self.out_file = out_file
        self.df = None
        self.metrics = {}

    def load_data(self):
        print(f"Loading data from {self.lead_file}")
        self.df = pd.read_csv(self.lead_file)
    
    def compute_basic_kpis(self):
        print("Computing basic KPIs...")
        total_leads = len(self.df)
        won_deals = self.df[self.df['deal_won'] == 1].shape[0]
        deal_win_rate = 100 * won_deals / total_leads if total_leads else 0
        avg_deal_size = self.df[self.df['deal_won'] == 1]['deal_size'].mean() if won_deals else 0

        self.metrics['Total Leads'] = total_leads
        self.metrics['Deals Won'] = won_deals
        self.metrics['Win Rate (%)'] = round(deal_win_rate,2)
        self.metrics['Avg. Won Deal Size'] = round(avg_deal_size,2)

    def compute_time_kpis(self):
        print("Computing time-based KPIs...")
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        this_month = datetime.date.today().month
        this_year = datetime.date.today().year
        recent = self.df[self.df['created_at'].dt.month == this_month]
        leads_this_month = recent.shape[0]
        won_this_month = recent[recent['deal_won'] == 1].shape[0]
        self.metrics['Leads This Month'] = leads_this_month
        self.metrics['Wins This Month'] = won_this_month

    def industry_breakdown(self, top_n=3):
        print("Calculating industry breakdown...")
        industries = self.df['industry'].value_counts().head(top_n)
        for idx, (ind, ct) in enumerate(industries.items()):
            self.metrics[f'Industry Top {idx+1}'] = f"{ind} ({ct})"

    def export_report(self):
        print(f"Exporting report to {self.out_file} ...")
        rep = pd.DataFrame(list(self.metrics.items()), columns=['Metric','Value'])
        os.makedirs(os.path.dirname(self.out_file), exist_ok=True)
        rep.to_csv(self.out_file, index=False)
        print("Done.")

    def print_report(self):
        print("\n*** KPI Executive Report ***")
        for k, v in self.metrics.items():
            print(f"{k:25}: {v}")
        print("****************************\n")

    def run(self):
        self.load_data()
        self.compute_basic_kpis()
        self.compute_time_kpis()
        self.industry_breakdown()
        self.export_report()
        self.print_report()

if __name__ == '__main__':
    reporter = KPIReporter()
    reporter.run()
