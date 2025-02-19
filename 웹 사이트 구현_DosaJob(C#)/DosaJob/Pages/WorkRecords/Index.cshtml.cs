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

namespace DosaJob.Pages.WorkRecords
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

        public PaginatedList<WorkRecord> WorkRecords { get; set; }

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

            IQueryable<WorkRecord> workRecordIQ = from s in _context.WorkRecords
                                                  join ct in _context.Categories on s.CategoryID equals ct.CategoryId into sct
                                                  from rs in sct.DefaultIfEmpty()
                                                  select s;
            

            if (!String.IsNullOrEmpty(searchString))
            {
                workRecordIQ = workRecordIQ.Where(s => s.StaffName.Contains(searchString)
                                       || s.Title.Contains(searchString));
            }
            switch (sortOrder)
            {
                case "date_asc":
                    workRecordIQ = workRecordIQ.OrderBy(s => s.CreatedDate);
                    break;
                case "Name":
                    workRecordIQ = workRecordIQ.OrderBy(s => s.StaffName);
                    break;
                case "name_desc":
                    workRecordIQ = workRecordIQ.OrderByDescending(s => s.StaffName);
                    break;
                default:
                    workRecordIQ = workRecordIQ.OrderByDescending(s => s.CreatedDate);
                    break;
            }

            var pageSize = Configuration.GetValue("PageSize", 20);
            WorkRecords = await PaginatedList<WorkRecord>.CreateAsync(
                workRecordIQ.AsNoTracking(), pageIndex ?? 1, pageSize);


            //WorkRecord = await _context.WorkRecord.ToListAsync();
        }
    }
}
