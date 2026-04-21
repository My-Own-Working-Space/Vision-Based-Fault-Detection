using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace FaultDetection.Web.Migrations
{
    /// <inheritdoc />
    public partial class InitialCreate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "InspectionLogs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    TowerId = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    FaultType = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    ConfidenceScore = table.Column<float>(type: "REAL", nullable: false),
                    ImagePath = table.Column<string>(type: "TEXT", nullable: false),
                    Latitude = table.Column<double>(type: "REAL", nullable: false),
                    Longitude = table.Column<double>(type: "REAL", nullable: false),
                    CreatedDate = table.Column<DateTime>(type: "TEXT", nullable: false),
                    IsRepaired = table.Column<bool>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_InspectionLogs", x => x.Id);
                });

            migrationBuilder.InsertData(
                table: "InspectionLogs",
                columns: new[] { "Id", "ConfidenceScore", "CreatedDate", "FaultType", "ImagePath", "IsRepaired", "Latitude", "Longitude", "TowerId" },
                values: new object[,]
                {
                    { 1, 0.98f, new DateTime(2026, 4, 21, 12, 58, 12, 360, DateTimeKind.Utc).AddTicks(3189), "Bình thường", "/images/sample_ok.jpg", false, 21.028500000000001, 105.85420000000001, "T-110KV-01" },
                    { 2, 0.85f, new DateTime(2026, 4, 21, 13, 58, 12, 360, DateTimeKind.Utc).AddTicks(3819), "Sứ nứt", "/images/sample_fault.jpg", false, 21.0305, 105.8522, "T-110KV-02" }
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "InspectionLogs");
        }
    }
}
