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

namespace DosaJob.Pages.WorkRecords
{
    public class CreateModel : PageModel
    {
        private readonly DosaJob.Data.DosaJobContext _context;

        public CreateModel(DosaJob.Data.DosaJobContext context)
        {
            _context = context;
        }

        public IActionResult OnGet()
        {
        ViewData["CategoryID"] = new SelectList(_context.Categories, "CategoryId", "CategoryName");
            return Page();
        }

        [BindProperty]
        public WorkRecord WorkRecord { get; set; }

        // To protect from overposting attacks, see https://aka.ms/RazorPagesCRUD
        public async Task<IActionResult> OnPostAsync()
        {
            if (!ModelState.IsValid)
            {
                return Page();
            }

            _context.WorkRecords.Add(WorkRecord);
            await _context.SaveChangesAsync();

            return RedirectToPage("./Index");
        }
    }
}
