using System.ComponentModel.DataAnnotations;

namespace DosaJob.Models
{
    public class WorkRecord
    {
        public int ID { get; set; }

        [Display(Name ="제목")]
        public string Title { get; set; } = string.Empty;

        [Display(Name ="대상직원")]
        public string StaffName { get; set; } = string.Empty;
        
        [DataType(DataType.DateTime)]
        public DateTime? CreatedDate { get; set; }

        [Display(Name = "내용")]
        public string Content { get; set; } = string.Empty;

        [Display(Name = "업무유형")]
        public int? CategoryID { get; set; }

        public Category? Category { get; set; }

    }
}
