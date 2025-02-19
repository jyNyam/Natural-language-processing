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
    public class DeleteModel : PageModel
    {
        private readonly DosaJob.Data.DosaJobContext _context;

        public DeleteModel(DosaJob.Data.DosaJobContext context)
        {
            _context = context;
        }

        [BindProperty]
        public WorkRecord WorkRecord { get; set; }

        public async Task<IActionResult> OnGetAsync(int? id)
        {
            if (id == null)
            {
                return NotFound();
            }

            WorkRecord = await _context.WorkRecords
                .Include(w => w.Category).FirstOrDefaultAsync(m => m.ID == id);

            if (WorkRecord == null)
            {
                return NotFound();
            }
            return Page();
        }

        public async Task<IActionResult> OnPostAsync(int? id)
        {
            if (id == null)
            {
                return NotFound();
            }

            WorkRecord = await _context.WorkRecords.FindAsync(id);

            if (WorkRecord != null)
            {
                _context.WorkRecords.Remove(WorkRecord);
                await _context.SaveChangesAsync();
            }

            return RedirectToPage("./Index");
        }
    }
}
