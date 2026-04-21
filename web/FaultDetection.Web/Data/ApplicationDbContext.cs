using FaultDetection.Web.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Diagnostics;

namespace FaultDetection.Web.Data
{
    public class ApplicationDbContext : DbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {
        }

        public DbSet<InspectionLog> InspectionLogs { get; set; }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.ConfigureWarnings(w => w.Ignore(RelationalEventId.PendingModelChangesWarning));
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);
            
            // Có thể thêm seed data ở đây nếu cần
            modelBuilder.Entity<InspectionLog>().HasData(
                new InspectionLog { Id = 1, TowerId = "T-110KV-01", FaultType = "Bình thường", ConfidenceScore = 0.98f, Latitude = 21.0285, Longitude = 105.8542, ImagePath = "/images/sample_ok.jpg", CreatedDate = DateTime.UtcNow.AddHours(-2) },
                new InspectionLog { Id = 2, TowerId = "T-110KV-02", FaultType = "Sứ nứt", ConfidenceScore = 0.85f, Latitude = 21.0305, Longitude = 105.8522, ImagePath = "/images/sample_fault.jpg", CreatedDate = DateTime.UtcNow.AddHours(-1) }
            );
        }
    }
}
