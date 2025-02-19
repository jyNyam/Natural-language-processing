#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using DosaJob.Models;

namespace DosaJob.Data
{
    public class DosaJobContext : DbContext
    {
        public DosaJobContext (DbContextOptions<DosaJobContext> options)
            : base(options)
        {
        }

        public DbSet<DosaJob.Models.WorkRecord> WorkRecords { get; set; }
        public DbSet<DosaJob.Models.Category> Categories  { get; set; }

        public DbSet<DosaJob.Models.WeeklyReport> WeeklyReports { get; set; }

    }
}
