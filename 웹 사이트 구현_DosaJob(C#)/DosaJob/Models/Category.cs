using System.ComponentModel.DataAnnotations;

namespace DosaJob.Models
{
    public class Category
    {
        public int CategoryId { get; set; }

        [Display(Name = "업무분류")]
        public string? CategoryName { get; set; }

        [Display(Name = "설명")]
        public String? Bigo { get; set;}

        ICollection<WorkRecord> WorkRecords { get; set; }
    }
}
