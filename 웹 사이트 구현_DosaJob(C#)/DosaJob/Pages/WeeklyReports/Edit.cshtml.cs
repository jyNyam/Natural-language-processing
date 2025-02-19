#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;
using DosaJob.Data;
using DosaJob.Models;

namespace DosaJob.Pages.WeeklyReports
{
    public class EditModel : PageModel
    {
        private readonly DosaJob.Data.DosaJobContext _context;

        public EditModel(DosaJob.Data.DosaJobContext context)
        {
            _context = context;
        }

        [BindProperty]
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

        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see https://aka.ms/RazorPagesCRUD.
        public async Task<IActionResult> OnPostAsync()
        {
            if (!ModelState.IsValid)
            {
                return Page();
            }

            _context.Attach(WeeklyReport).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!WeeklyReportExists(WeeklyReport.WeeklyReportID))
                {
                    return NotFound();
                }
                else
                {
                    throw;
                }
            }

            return RedirectToPage("./Index");
        }

        private bool WeeklyReportExists(int id)
        {
            return _context.WeeklyReports.Any(e => e.WeeklyReportID == id);
        }
    }
}
