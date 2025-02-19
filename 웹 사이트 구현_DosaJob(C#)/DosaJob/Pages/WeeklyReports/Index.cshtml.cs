#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.EntityFrameworkCore;
using DosaJob.Data;
using DosaJob.Models;

namespace DosaJob.Pages.WeeklyReports
{
    public class IndexModel : PageModel
    {
        private readonly DosaJob.Data.DosaJobContext _context;
        private readonly IConfiguration Configuration;

        public IndexModel(DosaJob.Data.DosaJobContext context, IConfiguration configuration)
        {
            _context = context;
            Configuration = configuration;
        }

        public PaginatedList<WeeklyReport> WeeklyReports { get;set; }
        public string NameSort { get; set; }
        public string DateSort { get; set; }
        public string CurrentFilter { get; set; }
        public string CurrentSort { get; set; }
        public async Task OnGetAsync(string sortOrder,
            string currentFilter, string searchString, int? pageIndex)
        {
            CurrentSort = sortOrder;
            //NameSort = String.IsNullOrEmpty(sortOrder) ? "name_desc" : "";
            //DateSort = sortOrder == "Date" ? "date_desc" : "Date";

            DateSort = String.IsNullOrEmpty(sortOrder) ? "date_asc" : "";
            NameSort = sortOrder == "Name" ? "name_desc" : "Name";

            if (searchString != null)
            {
                pageIndex = 1;
            }
            else
            {
                searchString = currentFilter;
            }

            CurrentFilter = searchString;

            IQueryable<WeeklyReport> workRecordIQ = from s in _context.WeeklyReports
                                                  select s;
            if (!String.IsNullOrEmpty(searchString))
            {
                workRecordIQ = workRecordIQ.Where(s => s.Title.Contains(searchString)
                                       || s.ThisWeek.Contains(searchString));
            }
            switch (sortOrder)
            {
                case "date_asc":
                    workRecordIQ = workRecordIQ.OrderBy(s => s.ReportDate);
                    break;
                case "Name":
                    workRecordIQ = workRecordIQ.OrderBy(s => s.Title);
                    break;
                case "name_desc":
                    workRecordIQ = workRecordIQ.OrderByDescending(s => s.Title);
                    break;
                default:
                    workRecordIQ = workRecordIQ.OrderByDescending(s => s.ReportDate);
                    break;
            }

            var pageSize = Configuration.GetValue("PageSize", 20);
            WeeklyReports = await PaginatedList<WeeklyReport>.CreateAsync(
                workRecordIQ.AsNoTracking(), pageIndex ?? 1, pageSize);

        }
    }
}
