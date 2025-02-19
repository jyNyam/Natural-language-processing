using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace DosaJob.Migrations
{
    public partial class CategoryTable : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropPrimaryKey(
                name: "PK_WorkRecord",
                table: "WorkRecord");

            migrationBuilder.RenameTable(
                name: "WorkRecord",
                newName: "WorkRecords");

            migrationBuilder.AddColumn<int>(
                name: "CategoryID",
                table: "WorkRecords",
                type: "int",
                nullable: true);

            migrationBuilder.AddPrimaryKey(
                name: "PK_WorkRecords",
                table: "WorkRecords",
                column: "ID");

            migrationBuilder.CreateTable(
                name: "Categories",
                columns: table => new
                {
                    CategoryId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    CategoryName = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    Bigo = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Categories", x => x.CategoryId);
                });

            migrationBuilder.CreateIndex(
                name: "IX_WorkRecords_CategoryID",
                table: "WorkRecords",
                column: "CategoryID");

            migrationBuilder.AddForeignKey(
                name: "FK_WorkRecords_Categories_CategoryID",
                table: "WorkRecords",
                column: "CategoryID",
                principalTable: "Categories",
                principalColumn: "CategoryId");
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_WorkRecords_Categories_CategoryID",
                table: "WorkRecords");

            migrationBuilder.DropTable(
                name: "Categories");

            migrationBuilder.DropPrimaryKey(
                name: "PK_WorkRecords",
                table: "WorkRecords");

            migrationBuilder.DropIndex(
                name: "IX_WorkRecords_CategoryID",
                table: "WorkRecords");

            migrationBuilder.DropColumn(
                name: "CategoryID",
                table: "WorkRecords");

            migrationBuilder.RenameTable(
                name: "WorkRecords",
                newName: "WorkRecord");

            migrationBuilder.AddPrimaryKey(
                name: "PK_WorkRecord",
                table: "WorkRecord",
                column: "ID");
        }
    }
}
