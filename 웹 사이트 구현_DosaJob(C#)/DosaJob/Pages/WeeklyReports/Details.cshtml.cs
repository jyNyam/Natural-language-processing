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
    public class DetailsModel : PageModel
    {
        private readonly DosaJob.Data.DosaJobContext _context;

        public DetailsModel(DosaJob.Data.DosaJobContext context)
        {
            _context = context;
        }

        public WeeklyReport WeeklyReport { get; set; }

        public async Task<IActionResult> OnGetAsync(int? id)
        {
            if (id == null)
            {
                return NotFound();
            }

            WeeklyReport = await _context.WeeklyReports.FirstOrDefaultAsync(m => m.WeeklyReportID == id);

            if (WeeklyReport == null)
            {
                return NotFound();
            }
            return Page();
        }
    }
}
