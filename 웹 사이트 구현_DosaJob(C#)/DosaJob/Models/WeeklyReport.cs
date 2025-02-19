using System.ComponentModel.DataAnnotations;

namespace DosaJob.Models
{
    public class WeeklyReport
    {
        public int WeeklyReportID { get; set; }

        [Display(Name = "제목")]
        public string Title { get; set; } = string.Empty;

        [DataType(DataType.DateTime)]
        public DateTime? ReportDate { get; set; }

        [Display(Name = "지난 주")]
        public string ThisWeek { get; set; } = string.Empty;

        [Display(Name = "이번 주")]
        public string NextWeek { get; set; } = string.Empty;

        [Display(Name = "비고")]        
        public string? Bigo { get; set; }
    }
}
