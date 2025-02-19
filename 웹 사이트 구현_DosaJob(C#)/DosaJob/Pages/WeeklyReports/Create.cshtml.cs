#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.AspNetCore.Mvc.Rendering;
using DosaJob.Data;
using DosaJob.Models;

namespace DosaJob.Pages.WeeklyReports
{
    public class CreateModel : PageModel
    {
        private readonly DosaJob.Data.DosaJobContext _context;
        public string strThisWeek = string.Empty;

        public CreateModel(DosaJob.Data.DosaJobContext context)
        {
            _context = context;
        }

        public IActionResult OnGet()
        {
            WeeklyReport weeklyReport = (from s in _context.WeeklyReports
                                               orderby s.ReportDate descending
                                               select s).FirstOrDefault();

            //ViewData["thisWeek"] = "";


            if (weeklyReport != null)
            {
                //ViewData["thisWeek"] = weeklyReport.NextWeek.ToString();
                this.strThisWeek = weeklyReport.NextWeek.ToString();
            }


            return Page();
        }

        [BindProperty]
        public WeeklyReport WeeklyReport { get; set; }

        // To protect from overposting attacks, see https://aka.ms/RazorPagesCRUD
        public async Task<IActionResult> OnPostAsync(string ThisWeek)
        {
            if (!ModelState.IsValid)
            {
                return Page();
            }

            WeeklyReport.ThisWeek = ThisWeek;
            _context.WeeklyReports.Add(WeeklyReport);
            await _context.SaveChangesAsync();

            return RedirectToPage("./Index");
        }
    }
}
