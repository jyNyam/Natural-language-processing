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

namespace DosaJob.Pages.Categories
{
    public class IndexModel : PageModel
    {
        private readonly DosaJob.Data.DosaJobContext _context;

        public IndexModel(DosaJob.Data.DosaJobContext context)
        {
            _context = context;
        }

        public IList<Category> Category { get;set; }

        public async Task OnGetAsync()
        {
            Category = await _context.Categories.ToListAsync();
        }
    }
}
