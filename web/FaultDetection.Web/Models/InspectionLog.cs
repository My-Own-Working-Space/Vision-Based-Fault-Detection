using System;
using System.ComponentModel.DataAnnotations;

namespace FaultDetection.Web.Models
{
    public class InspectionLog
    {
        public int Id { get; set; }

        [Required]
        [StringLength(50)]
        public string TowerId { get; set; } = string.Empty;

        [Required]
        [StringLength(50)]
        public string FaultType { get; set; } = string.Empty; // Sứ nứt, Rỉ sét, Bình thường

        public float ConfidenceScore { get; set; }

        [Required]
        public string ImagePath { get; set; } = string.Empty;

        public double Latitude { get; set; }
        public double Longitude { get; set; }

        public DateTime CreatedDate { get; set; } = DateTime.UtcNow;

        public bool IsRepaired { get; set; } = false;
    }
}
