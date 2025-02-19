using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace DosaJob.Migrations
{
    public partial class AddWeeklyReport : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "WeeklyReports",
                columns: table => new
                {
                    WeeklyReportID = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Title = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    ReportDate = table.Column<DateTime>(type: "datetime2", nullable: true),
                    ThisWeek = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    NextWeek = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Bigo = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_WeeklyReports", x => x.WeeklyReportID);
                });
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "WeeklyReports");
        }
    }
}
