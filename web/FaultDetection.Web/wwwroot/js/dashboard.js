document.addEventListener('DOMContentLoaded', function () {
    // Initialize Map
    var map = L.map('map').setView([21.0285, 105.8542], 13); // Hanoi coordinates

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    var markers = [];

    // Function to load faults
    async function loadFaults() {
        try {
            const response = await fetch('/api/faults');
            const data = await response.json();
            
            updateDashboard(data);
        } catch (error) {
            console.error('Error fetching faults:', error);
        }
    }

    function updateDashboard(data) {
        // Clear markers
        markers.forEach(m => map.removeLayer(m));
        markers = [];
        
        const rows = document.getElementById('fault-rows');
        const countBadge = document.getElementById('fault-count');
        
        rows.innerHTML = '';
        countBadge.innerText = data.filter(f => f.faultType !== 'Bình thường').length;

        data.forEach(fault => {
            // Add marker if location exists
            if (fault.latitude && fault.longitude) {
                var color = fault.faultType === 'Bình thường' ? 'green' : (fault.faultType === 'Sứ nứt' ? 'red' : 'orange');
                
                var markerIcon = L.circleMarker([fault.latitude, fault.longitude], {
                    radius: 8,
                    fillColor: color,
                    color: "#fff",
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                }).addTo(map);
                
                markerIcon.bindPopup(`<b>Tower: ${fault.towerId}</b><br>Fault: ${fault.faultType}<br>Conf: ${Math.round(fault.confidenceScore * 100)}%`);
                markers.push(markerIcon);
            }

            // Add row to table
            const row = `
                <tr>
                    <td>${fault.id}</td>
                    <td><strong>${fault.towerId}</strong></td>
                    <td>
                        <span class="badge ${getBadgeClass(fault.faultType)}">${fault.faultType}</span>
                    </td>
                    <td>${Math.round(fault.confidenceScore * 100)}%</td>
                    <td>${new Date(fault.createdDate).toLocaleString()}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary" onclick="viewDetail(${fault.id})">Chi tiết</button>
                    </td>
                </tr>
            `;
            rows.innerHTML += row;
        });
    }

    function getBadgeClass(type) {
        if (type === 'Bình thường') return 'bg-success';
        if (type === 'Sứ nứt') return 'bg-danger';
        if (type === 'Rỉ sét') return 'bg-warning text-dark';
        return 'bg-secondary';
    }

    window.viewDetail = function(id) {
        alert("Chức năng xem chi tiết ảnh đang được phát triển cho ID: " + id);
    };

    // Initial load
    loadFaults();

    // Refresh every 10 seconds
    setInterval(loadFaults, 10000);
});
